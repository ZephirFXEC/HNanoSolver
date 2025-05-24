#pragma once

#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

// Helper function to sample SDF value at a specific coordinate
template <typename Vec3T>
__device__ float sampleSDF(const float* sdfData, const Vec3T& coord, const IndexSampler<float, 1>& sampler) {
	if (!sdfData) return 1.0f;  // If no SDF data, return a value representing "outside" (no collision)
	return sampler(coord);
}


template <typename Vec3T>
__device__ nanovdb::Vec3f gradientSDF(const float* sdfData, const Vec3T& coord, const IndexSampler<float, 1>& sampler,
                                      const float inv_voxelSize) {
	if (!sdfData) return nanovdb::Vec3f(0);  // If no SDF data, return a value representing "outside" (no collision)

	const float right = sampler(coord + Vec3T(1, 0, 0));
	const float left = sampler(coord + Vec3T(-1, 0, 0));
	const float top = sampler(coord + Vec3T(0, 1, 0));
	const float bottom = sampler(coord + Vec3T(0, -1, 0));
	const float front = sampler(coord + Vec3T(0, 0, 1));
	const float back = sampler(coord + Vec3T(0, 0, -1));

	return nanovdb::Vec3f(right - left, top - bottom, front - back) * (0.5f * inv_voxelSize);
}

// Check if a position is inside a collision object
template <typename Vec3T>
__device__ bool isInCollision(const float* sdfData, const Vec3T& pos, const IndexSampler<float, 1>& sampler, float threshold) {
	if (!sdfData) return false;

	const float sdfValue = sampleSDF(sdfData, pos, sampler);
	return sdfValue < threshold;  // Negative inside, positive outside
}

// Compute gradient of SDF at a position to get normal vector
template <typename Vec3T>
__device__ nanovdb::Vec3f getSDFNormal(const float* sdfData, Vec3T& pos, const IndexSampler<float, 1>& sampler, float epsilon) {
	if (!sdfData) return nanovdb::Vec3f(0.0f);

	const nanovdb::Vec3f g = gradientSDF(sdfData, pos, sampler, epsilon);
	const float len = g.length();
	return len > 1e-6f ? g / len : nanovdb::Vec3f(0.0f);
}

__device__ nanovdb::Vec3f applySpecularReflection(const nanovdb::Vec3f& v,
                                                  const nanovdb::Vec3f& n)  // n must be normalised
{
	const float vdotn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
	return v - n * (2.0f * vdotn);  // subtract 2×(v·n) n
}

// Apply no-slip boundary condition
__device__ nanovdb::Vec3f applyNoSlipBoundary(const nanovdb::Vec3f& velocity, const nanovdb::Vec3f& normal) {
	// No-slip: remove all velocity at boundary
	// For a point near the boundary, we project the velocity onto the tangent plane
	// and then scale it based on distance to properly enforce no-slip

	// Project velocity onto normal
	const float vdotn = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];

	// Compute the normal component of the velocity
	const nanovdb::Vec3f v_normal = normal * vdotn;

	// Subtract normal component to get tangential component only
	const nanovdb::Vec3f v_tangent = velocity - v_normal;

	// For strict no-slip, even the tangential component should be zero at the boundary
	// We'll return zero velocity (completely stopped by obstacle)
	return v_tangent;  // nanovdb::Vec3f(0.0f);
}

// Kernel to enforce collision boundaries
__global__ void enforceCollisionBoundaries(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                           const nanovdb::Coord* __restrict__ coords, nanovdb::Vec3f* __restrict__ velocityData,
                                           const float* __restrict__ collisionSDF, const float voxelSize, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	if (!collisionSDF) return;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const IndexSampler<float, 1> sdfSampler(idxSampler, collisionSDF);
	const nanovdb::Coord coord = coords[idx];

	// Check if we're inside or close to a collision object
	const float sdf_value = sampleSDF(collisionSDF, coord, sdfSampler);

	// Inside collision: set velocity to zero
	if (sdf_value < 0.0f) {
		velocityData[idx] = nanovdb::Vec3f(0.0f);
		return;
	}

	// Within collision margin: apply boundary condition
	const float collisionMargin = 0.1;  // Margin in voxels
	if (sdf_value < collisionMargin) {
		// Get the boundary normal
		const nanovdb::Vec3f normal = getSDFNormal(collisionSDF, coord, sdfSampler, 1.0f / voxelSize);

		// Compute blend factor based on distance (closer to boundary = more damping)
		const float blend = 1.0f - (sdf_value / collisionMargin);

		// Get current velocity
		const nanovdb::Vec3f velocity = velocityData[idx];

		// Apply no-slip boundary condition
		const nanovdb::Vec3f modified_velocity = applyNoSlipBoundary(velocity, normal);

		// Blend between original and modified velocity based on distance to boundary
		velocityData[idx] = velocity * (1.0f - blend) + modified_velocity * blend;
	}
}

__global__ void advect_scalars(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                               const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                               float** __restrict__ inDataArrays, float** __restrict__ outDataArrays, const int numScalars,
                               const float* __restrict__ collisionSDF, const bool hasCollision, const size_t totalVoxels, const float dt,
                               const float inv_voxelSize) {
	IndexOffsetSampler<0> s_idxSampler(domainGrid);
	const IndexSampler<float, 1> sdfSampler(s_idxSampler, collisionSDF);

	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float scaled_dt = dt * inv_voxelSize;

	const nanovdb::Coord coord = coords[idx];
	uint64_t origIndex = s_idxSampler.offset(coord);
	origIndex = origIndex == 0 ? 0 : origIndex - 1;

	const nanovdb::Vec3f posCell = coord.asVec3s();
	const nanovdb::Vec3f velCenter = velocityData[origIndex];

	// Semi-Lagrangian backtrace
	nanovdb::Vec3f backPos = posCell - velCenter * scaled_dt;

	// Collision handling for backtracing
	if (hasCollision && collisionSDF) {
		// Check if backtraced position is in collision
		if (isInCollision(collisionSDF, backPos, sdfSampler, 0.0f)) {
			// If in collision, do not backtrace through object - use current position instead
			backPos = posCell;
		}
	}

	// Precompute interpolation data for backPos
	struct InterpData {
		uint64_t indices[8];
		float weights[8];
	};
	InterpData backPosData;
	{
		const float x = backPos[0], y = backPos[1], z = backPos[2];
		const int i0 = floor(x), i1 = i0 + 1;
		const int j0 = floor(y), j1 = j0 + 1;
		const int k0 = floor(z), k1 = k0 + 1;
		const float tx = x - i0, ty = y - j0, tz = z - k0;
		const float w000 = (1 - tx) * (1 - ty) * (1 - tz), w100 = tx * (1 - ty) * (1 - tz);
		const float w010 = (1 - tx) * ty * (1 - tz), w110 = tx * ty * (1 - tz);
		const float w001 = (1 - tx) * (1 - ty) * tz, w101 = tx * (1 - ty) * tz;
		const float w011 = (1 - tx) * ty * tz, w111 = tx * ty * tz;

		uint64_t indices[8] = {s_idxSampler.offset(nanovdb::Coord(i0, j0, k0)), s_idxSampler.offset(nanovdb::Coord(i1, j0, k0)),
		                       s_idxSampler.offset(nanovdb::Coord(i0, j1, k0)), s_idxSampler.offset(nanovdb::Coord(i1, j1, k0)),
		                       s_idxSampler.offset(nanovdb::Coord(i0, j0, k1)), s_idxSampler.offset(nanovdb::Coord(i1, j0, k1)),
		                       s_idxSampler.offset(nanovdb::Coord(i0, j1, k1)), s_idxSampler.offset(nanovdb::Coord(i1, j1, k1))};

		for (int j = 0; j < 8; ++j) {
			indices[j] = indices[j] == 0 ? 0 : indices[j] - 1;
		}

		backPosData = {{indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6], indices[7]},
		               {w000, w100, w010, w110, w001, w101, w011, w111}};
	}

	// Compute velF using precomputed backPosData
	nanovdb::Vec3f velF(0.0f);
#pragma unroll
	for (int j = 0; j < 8; ++j) {
		const nanovdb::Vec3f v = velocityData[backPosData.indices[j]];
		velF += v * backPosData.weights[j];
	}

	nanovdb::Vec3f fwdPos2 = backPos + velF * scaled_dt;

	// Check for collision in the forward trace as well
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, fwdPos2, sdfSampler, 0.0f)) {
			fwdPos2 = backPos;
		}
	}

	// Precompute interpolation data for fwdPos2
	InterpData fwdPos2Data;
	{
		const float x = fwdPos2[0], y = fwdPos2[1], z = fwdPos2[2];
		const int i0 = floor(x), i1 = i0 + 1;
		const int j0 = floor(y), j1 = j0 + 1;
		const int k0 = floor(z), k1 = k0 + 1;
		const float tx = x - i0, ty = y - j0, tz = z - k0;
		const float w000 = (1 - tx) * (1 - ty) * (1 - tz), w100 = tx * (1 - ty) * (1 - tz);
		const float w010 = (1 - tx) * ty * (1 - tz), w110 = tx * ty * (1 - tz);
		const float w001 = (1 - tx) * (1 - ty) * tz, w101 = tx * (1 - ty) * tz;
		const float w011 = (1 - tx) * ty * tz, w111 = tx * ty * tz;

		uint64_t indices[8] = {s_idxSampler.offset(nanovdb::Coord(i0, j0, k0)), s_idxSampler.offset(nanovdb::Coord(i1, j0, k0)),
		                       s_idxSampler.offset(nanovdb::Coord(i0, j1, k0)), s_idxSampler.offset(nanovdb::Coord(i1, j1, k0)),
		                       s_idxSampler.offset(nanovdb::Coord(i0, j0, k1)), s_idxSampler.offset(nanovdb::Coord(i1, j0, k1)),
		                       s_idxSampler.offset(nanovdb::Coord(i0, j1, k1)), s_idxSampler.offset(nanovdb::Coord(i1, j1, k1))};

		for (int j = 0; j < 8; ++j) {
			indices[j] = indices[j] == 0 ? 0 : indices[j] - 1;
		}

		fwdPos2Data = {{indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6], indices[7]},
		               {w000, w100, w010, w110, w001, w101, w011, w111}};
	}

	// Precompute neighbor indices for clamp step
	static __device__ const int offs[6][3] = {{-1, 0, 0}, {+1, 0, 0}, {0, -1, 0}, {0, +1, 0}, {0, 0, -1}, {0, 0, +1}};
	uint32_t nbrIdx[6];
#pragma unroll
	for (int n = 0; n < 6; ++n) {
		nbrIdx[n] = s_idxSampler.offset(coord[0] + offs[n][0], coord[1] + offs[n][1], coord[2] + offs[n][2]);
	}
#pragma unroll
	for (int n = 0; n < 6; ++n) {
		nbrIdx[n] = nbrIdx[n] == 0 ? 0 : nbrIdx[n] - 1;
	}

	// Process each scalar
	for (int s = 0; s < numScalars; ++s) {
		const float* __restrict__ inData = inDataArrays[s];
		float* __restrict__ outData = outDataArrays[s];

		const float phiOrig = inData[origIndex];

		// Compute phiForward with unrolled interpolation
		float phiForward =
		    inData[backPosData.indices[0]] * backPosData.weights[0] + inData[backPosData.indices[1]] * backPosData.weights[1] +
		    inData[backPosData.indices[2]] * backPosData.weights[2] + inData[backPosData.indices[3]] * backPosData.weights[3] +
		    inData[backPosData.indices[4]] * backPosData.weights[4] + inData[backPosData.indices[5]] * backPosData.weights[5] +
		    inData[backPosData.indices[6]] * backPosData.weights[6] + inData[backPosData.indices[7]] * backPosData.weights[7];

		// Compute phiBackward with unrolled interpolation
		float phiBackward =
		    inData[fwdPos2Data.indices[0]] * fwdPos2Data.weights[0] + inData[fwdPos2Data.indices[1]] * fwdPos2Data.weights[1] +
		    inData[fwdPos2Data.indices[2]] * fwdPos2Data.weights[2] + inData[fwdPos2Data.indices[3]] * fwdPos2Data.weights[3] +
		    inData[fwdPos2Data.indices[4]] * fwdPos2Data.weights[4] + inData[fwdPos2Data.indices[5]] * fwdPos2Data.weights[5] +
		    inData[fwdPos2Data.indices[6]] * fwdPos2Data.weights[6] + inData[fwdPos2Data.indices[7]] * fwdPos2Data.weights[7];

		// Correction step
		const float error = phiOrig - phiBackward;
		float phiCorr = phiForward + 0.5f * error;

		// Clamp step
		float minVal = phiOrig, maxVal = phiOrig;
		for (int neighborIndice : nbrIdx) {
			const float val = inData[neighborIndice];
			minVal = fminf(minVal, val);
			maxVal = fmaxf(maxVal, val);
		}
		minVal = fminf(minVal, phiForward);
		maxVal = fmaxf(maxVal, phiForward);
		phiCorr = fmaxf(minVal, fminf(phiCorr, maxVal));

		outData[idx] = phiCorr;
	}
}


__global__ void advect_scalar(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              const float* __restrict__ inData, float* __restrict__ outData, const float* __restrict__ collisionSDF,
                              const bool hasCollision, const size_t totalVoxels, const float dt, const float inv_voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto sdfSampler = IndexSampler<float, 1>(idxSampler, collisionSDF);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto dataSampler = IndexSampler<float, 1>(idxSampler, inData);

	// The cell coordinate
	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f posCell = coord.asVec3s();
	const float phiOrig = dataSampler(coord);

	// MAC Velocity for backtrace
	const nanovdb::Vec3f velCenter = velocitySampler(coord);

	// ------------------------------------------------------------------------
	// Forward pass (semi-Lagrangian backtrace)
	// x_forward = x - vel*dt
	nanovdb::Vec3f backPos = posCell - velCenter * scaled_dt;

	// Check for collisions during backtracing
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, backPos, sdfSampler, 0.0f)) {
			backPos = posCell;
		}
	}

	const float phiForward = dataSampler(backPos);

	// ------------------------------------------------------------------------
	// Backward pass
	// x_backward = x_forward + u(x_forward)*dt
	// Then compare that to the original value
	const nanovdb::Vec3f velF = velocitySampler(backPos);
	nanovdb::Vec3f fwdPos2 = backPos + velF * scaled_dt;

	// Check for collisions during forward tracing
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, fwdPos2, sdfSampler, 0.0f)) {
			fwdPos2 = backPos;
		}
	}

	const float phiBackward = dataSampler(fwdPos2);

	// ------------------------------------------------------------------------
	// Correction
	// error = phiOrig - phiBackward
	// phiCorr = phiForward + 0.5 * error
	const float error = phiOrig - phiBackward;
	float phiCorr = phiForward + 0.5f * error;

	// ------------------------------------------------------------------------
	// Find local min/max in neighborhood for clamping
	float minVal = phiOrig;
	float maxVal = phiOrig;

	// Check 6-neighborhood for min/max values
	for (int dim = 0; dim < 3; ++dim) {
		for (int offset = -1; offset <= 1; offset += 2) {
			nanovdb::Coord neighborCoord = coord;
			neighborCoord[dim] += offset;
			const float neighborVal = dataSampler(neighborCoord);
			minVal = fminf(minVal, neighborVal);
			maxVal = fmaxf(maxVal, neighborVal);
		}
	}

	// Also include the semi-Lagrangian value in min/max computation
	minVal = fminf(minVal, phiForward);
	maxVal = fmaxf(maxVal, phiForward);

	// Clamp the result
	phiCorr = fmaxf(minVal, fminf(phiCorr, maxVal));

	outData[idx] = phiCorr;
}

__global__ void advect_vector(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              nanovdb::Vec3f* __restrict__ outVelocity, const float* __restrict__ collisionSDF, const bool hasCollision,
                              const size_t totalVoxels, const float dt, const float inv_voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto sdfSampler = IndexSampler<float, 1>(idxSampler, collisionSDF);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f pos = coord.asVec3s();

	// Original velocity at the cell or face
	const nanovdb::Vec3f velOrig = velocitySampler(coord);

	// Forward pass (backtrace)
	nanovdb::Vec3f backPos = pos - velOrig * scaled_dt;

	// Check for collisions during backtracing
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, backPos, sdfSampler, 0.0f)) {
			// If we hit a collision, use the original position
			backPos = pos;
		}
	}

	const nanovdb::Vec3f velForward = velocitySampler(backPos);

	// Backward check
	nanovdb::Vec3f fwdPos2 = backPos + velForward * scaled_dt;

	// Check for collisions during forward tracing
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, fwdPos2, sdfSampler, 0.0f)) {
			fwdPos2 = backPos;
		}
	}

	const nanovdb::Vec3f velBackward = velocitySampler(fwdPos2);

	// Correction
	const nanovdb::Vec3f errorVec = velOrig - velBackward;
	nanovdb::Vec3f velCorr = velForward + 0.5f * errorVec;

	// Find neighborhood min/max for each component
	nanovdb::Vec3f minVel, maxVel;
	for (int c = 0; c < 3; ++c) {
		minVel[c] = velOrig[c];
		maxVel[c] = velOrig[c];
	}

	// Check 6-neighborhood for min/max values
	for (int dim = 0; dim < 3; ++dim) {
		for (int offset = -1; offset <= 1; offset += 2) {
			nanovdb::Coord neighborCoord = coord;
			neighborCoord[dim] += offset;

			const nanovdb::Vec3f neighborVel = velocitySampler(neighborCoord);
			for (int c = 0; c < 3; ++c) {
				minVel[c] = fminf(minVel[c], neighborVel[c]);
				maxVel[c] = fmaxf(maxVel[c], neighborVel[c]);
			}
		}
	}

	// Also include the semi-Lagrangian value in min/max computation
	for (int c = 0; c < 3; ++c) {
		minVel[c] = fminf(minVel[c], velForward[c]);
		maxVel[c] = fmaxf(maxVel[c], velForward[c]);

		// Clamp the result
		velCorr[c] = fmaxf(minVel[c], fminf(velCorr[c], maxVel[c]));
	}

	// Handle collision boundaries for velocity
	if (hasCollision && collisionSDF) {
		const float sdf_value = sampleSDF(collisionSDF, coord, sdfSampler);

		if (sdf_value < 0.0f) {
			// Inside collision - zero velocity
			velCorr = nanovdb::Vec3f(0.0f);
		} else if (sdf_value < 0.1f) {
			// Near collision - apply no-slip boundary
			const nanovdb::Vec3f normal = getSDFNormal(collisionSDF, coord, sdfSampler, inv_voxelSize);
			const float blend = 1.0f - (sdf_value / 1.5f);

			// Calculate no-slip velocity
			const nanovdb::Vec3f no_slip = applyNoSlipBoundary(velCorr, normal);

			// Blend based on distance
			velCorr = velCorr * (1.0f - blend) + no_slip * blend;
		}
	}

	outVelocity[idx] = velCorr;
}

__global__ void divergence_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Vec3f* velocityData,
                               float* outDivergence, const float inv_dx, const int numLeaves) {
	// Block dimensions matching leaf size
	constexpr int BLOCK_SIZE = 8;

	const int leafIdx = blockIdx.x;

	if (leafIdx >= numLeaves) return;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;

	const auto& leaf = domainGrid->tree().getFirstNode<0>()[leafIdx];
	const nanovdb::Coord origin = leaf.origin();

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocityData);

	// Compute pressure gradient and update velocity
	if (tidx < BLOCK_SIZE && tidy < BLOCK_SIZE && tidz < BLOCK_SIZE) {
		const nanovdb::Coord coord = origin + nanovdb::Coord(tidx, tidy, tidz);

		const nanovdb::Vec3f current = velocitySampler(coord);

		// Average neighboring velocities for each component
		const float xp = 0.5f * (current[0] + velocitySampler(coord + nanovdb::Coord(1, 0, 0))[0]);
		const float xm = 0.5f * (current[0] + velocitySampler(coord - nanovdb::Coord(1, 0, 0))[0]);

		const float yp = 0.5f * (current[1] + velocitySampler(coord + nanovdb::Coord(0, 1, 0))[1]);
		const float ym = 0.5f * (current[1] + velocitySampler(coord - nanovdb::Coord(0, 1, 0))[1]);

		const float zp = 0.5f * (current[2] + velocitySampler(coord + nanovdb::Coord(0, 0, 1))[2]);
		const float zm = 0.5f * (current[2] + velocitySampler(coord - nanovdb::Coord(0, 0, 1))[2]);

		const float divergence = (xp - xm + yp - ym + zp - zm) * inv_dx;

		const auto idx = idxSampler.offset(coord);
		const auto cidx = idx == 0 ? 0 : idx - 1;
		outDivergence[cidx] = divergence;
	}
}


__global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                           const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                           float* __restrict__ outDivergence, const float inv_dx, const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Vec3f center = velocityData[tid];

	const nanovdb::Coord c = d_coord[tid];
	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocityData);

	const float xp = (center + velocitySampler(c + nanovdb::Coord(1, 0, 0)))[0] * 0.5f;
	const float xm = (center + velocitySampler(c - nanovdb::Coord(1, 0, 0)))[0] * 0.5f;
	const float yp = (center + velocitySampler(c + nanovdb::Coord(0, 1, 0)))[1] * 0.5f;
	const float ym = (center + velocitySampler(c - nanovdb::Coord(0, 1, 0)))[1] * 0.5f;
	const float zp = (center + velocitySampler(c + nanovdb::Coord(0, 0, 1)))[2] * 0.5f;
	const float zm = (center + velocitySampler(c - nanovdb::Coord(0, 0, 1)))[2] * 0.5f;

	outDivergence[tid] = (xp - xm + yp - ym + zp - zm) * inv_dx;
}

__global__ void redBlackGaussSeidelUpdate_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const float* divergence,
                                              float* pressure, const float dx, const size_t totalVoxels, const int color,
                                              const float omega) {
	constexpr int BLOCK_SIZE = 8;
	__shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	const int leafIdx = blockIdx.x;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;

	if (leafIdx >= domainGrid->tree().nodeCount(0)) return;

	const auto& leaf = domainGrid->tree().getFirstNode<0>()[leafIdx];
	const nanovdb::Coord origin = leaf.origin();
	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto pSampler = IndexSampler<float, 0>(idxSampler, pressure);

	// Load pressure halo (+1 in all directions)
	for (int idx = tidz * BLOCK_SIZE * BLOCK_SIZE + tidy * BLOCK_SIZE + tidx; idx < (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2);
	     idx += BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) {
		const int dz = idx / ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2));
		const int dy = (idx % ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))) / (BLOCK_SIZE + 2);
		const int dx = idx % (BLOCK_SIZE + 2);

		const nanovdb::Coord coord(origin.x() + dx - 1, origin.y() + dy - 1, origin.z() + dz - 1);
		s_pressure[dz][dy][dx] = pSampler(coord);
	}

	__syncthreads();

	// Process center region
	if (tidx < BLOCK_SIZE && tidy < BLOCK_SIZE && tidz < BLOCK_SIZE) {
		const nanovdb::Coord coord = origin + nanovdb::Coord(tidx, tidy, tidz);
		const int i = coord.x(), j = coord.y(), k = coord.z();

		// Red/black check
		if (((i + j + k) & 1) != color) return;

		// Find linear index in d_coord array
		size_t tid = idxSampler.offset(coord);
		tid = tid == 0 ? 0 : tid - 1;

		if (tid >= totalVoxels) return;

		// Shared memory indices with halo offset
		const int lx = tidx + 1;
		const int ly = tidy + 1;
		const int lz = tidz + 1;

		// Stencil accesses from shared memory
		const float pxp1 = s_pressure[lz][ly][lx + 1];
		const float pxm1 = s_pressure[lz][ly][lx - 1];
		const float pyp1 = s_pressure[lz][ly + 1][lx];
		const float pym1 = s_pressure[lz][ly - 1][lx];
		const float pzp1 = s_pressure[lz + 1][ly][lx];
		const float pzm1 = s_pressure[lz - 1][ly][lx];

		// Gauss-Seidel update
		const float dx2 = dx * dx;
		constexpr float inv6 = 0.166666667f;
		const float divVal = divergence[tid];
		const float pOld = s_pressure[lz][ly][lx];  // From shared memory

		const float pGS = ((pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1) - divVal * dx2) * inv6;
		pressure[tid] = pOld + omega * (pGS - pOld);
	}
}


__global__ void redBlackGaussSeidelUpdate(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                          const nanovdb::Coord* __restrict__ d_coord, const float* __restrict__ divergence,
                                          float* __restrict__ pressure, const float dx, const size_t totalVoxels, const int color,
                                          const float omega) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Coord c = d_coord[tid];
	const int i = c.x(), j = c.y(), k = c.z();

	// Skip if wrong color
	if (((i + j + k) & 1) != color) return;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto pSampler = IndexSampler<float, 0>(idxSampler, pressure);

	// Pre-compute common factors
	const float dx2 = dx * dx;
	constexpr float inv6 = 0.166666667;

	const float pxp1 = pSampler(nanovdb::Coord(i + 1, j, k));
	const float pxm1 = pSampler(nanovdb::Coord(i - 1, j, k));
	const float pyp1 = pSampler(nanovdb::Coord(i, j + 1, k));
	const float pym1 = pSampler(nanovdb::Coord(i, j - 1, k));
	const float pzp1 = pSampler(nanovdb::Coord(i, j, k + 1));
	const float pzm1 = pSampler(nanovdb::Coord(i, j, k - 1));

	const float divVal = divergence[tid];
	const float pOld = pressure[tid];

	const float pGS = ((pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1) - divVal * dx2) * inv6;
	pressure[tid] = pOld + omega * (pGS - pOld);
}

__global__ void restrict_to_4x4x4(const float* inData, float* outData, const size_t totalVoxels) {
	// totalVoxels should be 64 (for a 4x4x4 coarse grid)
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	// Map tid [0,63] to coarse grid coordinates in a 4x4x4 block:
	const int ic = tid % 4;
	const int jc = (tid / 4) % 4;
	const int kc = tid / 16;

	// Each coarse cell covers a 2x2x2 block in the fine grid.
	// Fine grid is 8x8x8; compute starting indices for the corresponding block.
	const int i_fine_start = ic * 2;
	const int j_fine_start = jc * 2;
	const int k_fine_start = kc * 2;

	float sum = 0.0f;
	// Loop over the 2x2x2 block
	for (int dz = 0; dz < 2; ++dz) {
		for (int dy = 0; dy < 2; ++dy) {
			for (int dx = 0; dx < 2; ++dx) {
				int i_fine = i_fine_start + dx;
				int j_fine = j_fine_start + dy;
				int k_fine = k_fine_start + dz;
				// Assuming fine grid is stored in x-fastest order:
				// Index = i + j * (8) + k * (8*8) where grid dimensions are 8×8×8.
				const int index = i_fine + j_fine * 8 + k_fine * 64;
				sum += inData[index];
			}
		}
	}
	// Average the sum of 8 fine cells
	outData[tid] = sum / 8.0f;
}


__global__ void redBlackGaussSeidelUpdate_single(const IndexOffsetSampler<0>& sampler, const nanovdb::Coord* d_coord,
                                                 const float* divergence, float* pressure, const float dx, const size_t totalVoxels,
                                                 const float omega, const int color) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Coord c = d_coord[tid];
	const int i = c.x(), j = c.y(), k = c.z();

	// Skip if wrong color
	if (((i + j + k) & 1) != color) return;

	const auto pSampler = IndexSampler<float, 0>(sampler, pressure);

	// Pre-compute common factors
	const float dx2 = dx * dx;
	constexpr float inv6 = 0.166666667;

	const float pxp1 = pSampler(nanovdb::Coord(i + 1, j, k));
	const float pxm1 = pSampler(nanovdb::Coord(i - 1, j, k));
	const float pyp1 = pSampler(nanovdb::Coord(i, j + 1, k));
	const float pym1 = pSampler(nanovdb::Coord(i, j - 1, k));
	const float pzp1 = pSampler(nanovdb::Coord(i, j, k + 1));
	const float pzm1 = pSampler(nanovdb::Coord(i, j, k - 1));

	const float divVal = divergence[tid];
	const float pOld = pressure[tid];

	const float pGS = ((pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1) - divVal * dx2) * inv6;
	pressure[tid] = pOld + omega * (pGS - pOld);
}


__global__ void subtractPressureGradient_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                             const nanovdb::Vec3f* __restrict__ velocity, const float* __restrict__ pressure,
                                             nanovdb::Vec3f* __restrict__ out, const float inv_voxelSize, const size_t numLeaves) {
	constexpr int BLOCK_SIZE = 8;
	__shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	const int leafIdx = blockIdx.x;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;

	if (leafIdx >= numLeaves) return;

	const auto& leaf = domainGrid->tree().getFirstNode<0>()[leafIdx];
	const nanovdb::Coord origin = leaf.origin();

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocity);
	const auto pressureSampler = IndexSampler<float, 0>(idxSampler, pressure);

	for (int idx = tidz * BLOCK_SIZE * BLOCK_SIZE + tidy * BLOCK_SIZE + tidx; idx < (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2);
	     idx += BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) {
		const int dz = idx / ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2));
		const int dy = (idx % ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))) / (BLOCK_SIZE + 2);
		const int dx = idx % (BLOCK_SIZE + 2);

		const nanovdb::Coord coord(origin.x() + dx - 1, origin.y() + dy - 1, origin.z() + dz - 1);
		s_pressure[dz][dy][dx] = pressureSampler(coord);
	}

	__syncthreads();

	if (tidx < BLOCK_SIZE && tidy < BLOCK_SIZE && tidz < BLOCK_SIZE) {
		const int lx = tidx + 1;
		const int ly = tidy + 1;
		const int lz = tidz + 1;

		const nanovdb::Coord coord = origin + nanovdb::Coord(tidx, tidy, tidz);

		// Load current cell's
		const nanovdb::Vec3f& u_star_c = velocitySampler(coord);

		// Pressure values at neighbours
		const float p_xp = s_pressure[lz][ly][lx + 1];  // p(i+1, j, k)
		const float p_xm = s_pressure[lz][ly][lx - 1];  // p(i-1, j, k)
		const float p_yp = s_pressure[lz][ly + 1][lx];  // p(i, j+1, k)
		const float p_ym = s_pressure[lz][ly - 1][lx];  // p(i, j-1, k)
		const float p_zp = s_pressure[lz + 1][ly][lx];  // p(i, j, k+1)
		const float p_zm = s_pressure[lz - 1][ly][lx];  // p(i, j, k-1)

		// Central difference gradient: grad(p)_x = (p(i+1) - p(i-1)) / (2*dx)
		// Multiply by 0.5f * inv_voxelSize which is 1 / (2 * dx)
		const float gradP_x = (p_xp - p_xm) * 0.5f * inv_voxelSize;
		const float gradP_y = (p_yp - p_ym) * 0.5f * inv_voxelSize;
		const float gradP_z = (p_zp - p_zm) * 0.5f * inv_voxelSize;

		// Form the gradient vector at the cell center
		const nanovdb::Vec3f gradP_c = {gradP_x, gradP_y, gradP_z};

		// --- Apply Pressure Gradient ---
		// u_n+1 = u* - dt * grad(p)  (Assuming rho=1)
		// Apply the update using the time step dt
		const nanovdb::Vec3f u_final_c = u_star_c - gradP_c;

		auto idx = idxSampler.offset(coord);
		idx = idx == 0 ? 0 : idx - 1;
		out[idx] = u_final_c;
	}
}


__global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                         const size_t totalVoxels, const nanovdb::Vec3f* velocity, const float* pressure,
                                         nanovdb::Vec3f* out, const float* __restrict__ collisionSDF, const bool hasCollision,
                                         const float inv_voxelSize) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	// Samplers
	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto pressureSampler = IndexSampler<float, 0>(idxSampler, pressure);
	const auto sdfSampler = IndexSampler<float, 1>(idxSampler, collisionSDF);

	// The cell center coordinate associated with this thread
	const nanovdb::Coord c = d_coords[tid];
	const nanovdb::Vec3f pos = c.asVec3s();

	// Get the intermediate cell-centered velocity u*
	const nanovdb::Vec3f u_star_c = velocity[tid];

	// --- Calculate Pressure Gradient at Cell Center (i, j, k) using Central Differences ---

	// Pressure values at neighbours
	const float p_xp = pressureSampler(c + nanovdb::Coord(1, 0, 0));  // p(i+1, j, k)
	const float p_xm = pressureSampler(c - nanovdb::Coord(1, 0, 0));  // p(i-1, j, k)
	const float p_yp = pressureSampler(c + nanovdb::Coord(0, 1, 0));  // p(i, j+1, k)
	const float p_ym = pressureSampler(c - nanovdb::Coord(0, 1, 0));  // p(i, j-1, k)
	const float p_zp = pressureSampler(c + nanovdb::Coord(0, 0, 1));  // p(i, j, k+1)
	const float p_zm = pressureSampler(c - nanovdb::Coord(0, 0, 1));  // p(i, j, k-1)

	// Central difference gradient: grad(p)_x = (p(i+1) - p(i-1)) / (2*dx)
	// Multiply by 0.5f * inv_voxelSize which is 1 / (2 * dx)
	const float gradP_x = (p_xp - p_xm);
	const float gradP_y = (p_yp - p_ym);
	const float gradP_z = (p_zp - p_zm);

	// Form the gradient vector at the cell center
	const nanovdb::Vec3f gradP_c = nanovdb::Vec3f(gradP_x, gradP_y, gradP_z) * 0.5f * inv_voxelSize;

	// --- Apply Pressure Gradient ---
	// u_n+1 = u* - dt * grad(p)
	// Apply the update using the time step dt
	nanovdb::Vec3f u_final_c = u_star_c - gradP_c;

	// Handle collision boundaries
	if (hasCollision && collisionSDF) {
		const float sdf_value = sampleSDF(collisionSDF, c, sdfSampler);

		if (sdf_value < 0.0f) {
			// Inside collision - zero velocity
			u_final_c = nanovdb::Vec3f(0.0f);
		} else if (sdf_value < 0.1f) {
			// Near collision - apply no-slip boundary
			const nanovdb::Vec3f normal = getSDFNormal(collisionSDF, c, sdfSampler, inv_voxelSize);
			const float blend = 1.0f - (sdf_value / 0.1f);

			// Calculate no-slip velocity
			const nanovdb::Vec3f no_slip = applyNoSlipBoundary(u_final_c, normal);

			// Blend based on distance
			u_final_c = u_final_c * (1.0f - blend) + no_slip * blend;
		}
	}

	out[tid] = u_final_c;
}

__global__ void temperature_buoyancy(const nanovdb::Vec3f* velocityData, const float* tempData, nanovdb::Vec3f* outVel, const float dt,
                                     const float ambient_temp, const float buoyancy_strength, const size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const nanovdb::Vec3f vel = velocityData[idx];
	const float temp = tempData[idx];
	if (temp <= ambient_temp) {
		outVel[idx] = vel;
		return;
	}

	const float tempDiff = temp - ambient_temp;
	const nanovdb::Vec3f buoyancyForce(0.0f, fmaxf(0.0f, tempDiff * buoyancy_strength), 0.0f);

	outVel[idx] = vel + buoyancyForce * dt;
}

__global__ void combustion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                           const float* __restrict__ fuelData, const float* __restrict__ tempData, float* __restrict__ outFuel,
                           float* __restrict__ outTemp, const float dt, float ignition_temp, float combustion_rate, float heat_release,
                           size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;


	const float fuel = fuelData[idx];
	const float temp = tempData[idx];
	float newFuel = fuel;
	float newTemp = temp;

	if (fuel > 0.0f && temp >= ignition_temp) {
		const float fuelBurned = fminf(fuel, combustion_rate * dt);
		newFuel -= fuelBurned;
		newTemp += fuelBurned * heat_release;
	}

	outFuel[idx] = newFuel;
	outTemp[idx] = newTemp;
}


__global__ void diffusion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                          const float* tempData, const float* fuelData, float* outTemp, float* outFuel, const float dt, float temp_diff,
                          float fuel_diff, float ambient_temp, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const IndexOffsetSampler<0> sampler(domainGrid);
	const auto tempSampler = IndexSampler<float, 0>(sampler, tempData);
	const auto fuelSampler = IndexSampler<float, 0>(sampler, fuelData);

	const nanovdb::Coord coord = d_coords[idx];
	const float centerTemp = tempSampler(coord);
	const float centerFuel = fuelSampler(coord);
	float tempLaplacian = 0.0f;
	float fuelLaplacian = 0.0f;
	int neighbors = 0;

	// Check each of the 6 direct neighbors
	nanovdb::Coord offsets[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

	for (auto offset : offsets) {
		const nanovdb::Coord neighborCoord = coord + offset;

		// Get the neighbor's index and values
		const float neighborTemp = tempSampler(neighborCoord);
		if (neighborTemp == 0.0f) continue;  // Skip inactive voxels

		const float neighborFuel = fuelSampler(neighborCoord);
		if (neighborFuel == 0.0f) continue;  // Skip inactive voxels

		// Accumulate Laplacian (difference from center to neighbor)
		tempLaplacian += (neighborTemp - centerTemp);
		fuelLaplacian += (neighborFuel - centerFuel);
		neighbors++;
	}

	// Apply diffusion if we have valid neighbors
	if (neighbors > 0) {
		outTemp[idx] = centerTemp + temp_diff * dt * tempLaplacian;
		outFuel[idx] = centerFuel + fuel_diff * dt * fuelLaplacian;
	} else {
		outTemp[idx] = centerTemp;
		outFuel[idx] = centerFuel;
	}

	// Apply cooling effect
	outTemp[idx] += (ambient_temp - outTemp[idx]) * dt * 0.1f;
}


__global__ void combustion_oxygen(const float* fuelData, const float* wasteData, const float* temperatureData, float* divergenceData,
                                  const float* flameData, float* outFuel, float* outWaste, float* outTemperature, float* outFlame,
                                  const float temp_gain, const float expansion, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	// Load input values for the current voxel
	float fuel = fuelData[idx];
	float waste = wasteData[idx];
	float temperature = temperatureData[idx];
	float flame = flameData[idx];

	// Apply fuel threshold
	if (fuel < 0.001f) {
		fuel = 0.0f;
	}

	// Calculate available oxygen
	float oxygen = 1.0f - fuel - waste;
	if (oxygen < 0.0f) {
		// Invalid state; copy inputs to outputs
		outFuel[idx] = fuel;
		outWaste[idx] = waste;
		outTemperature[idx] = temperature;
		outFlame[idx] = flame;
		return;
	}

	// Calculate burn amount (oxygen-limited, scaled by ratio)
	float burn = fminf(oxygen, fuel);

	// Update fields
	float newFuel = fuel - burn;
	float newWaste = waste + burn * 2.0f;                      // Fuel + oxygen consumed
	float newFlame = fmaxf(flame, fminf(1.0f, burn * 10.0f));  // Flame intensity
	float newTemperature = temperature + burn * temp_gain;

	// Write updated values to output arrays
	outFuel[idx] = newFuel;
	outWaste[idx] = newWaste;
	outTemperature[idx] = newTemperature;
	divergenceData[idx] += burn * expansion;
	outFlame[idx] = newFlame;
}


// Vorticity Confinement Kernel
__global__ void vorticityConfinement(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                     const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                                     nanovdb::Vec3f* __restrict__ outForce, const float dt, const float inv_dx,
                                     const float confinementScale, const float factorScale, const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Coord c = d_coord[tid];

	// Initialize sampler with boundary checks
	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocityData);
	const float factor = 0.5 * inv_dx;

	// Compute current cell's vorticity vector
	const nanovdb::Vec3f u_pX = velocitySampler(c + nanovdb::Coord(1, 0, 0));
	const nanovdb::Vec3f u_mX = velocitySampler(c - nanovdb::Coord(1, 0, 0));
	const nanovdb::Vec3f u_pY = velocitySampler(c + nanovdb::Coord(0, 1, 0));
	const nanovdb::Vec3f u_mY = velocitySampler(c - nanovdb::Coord(0, 1, 0));
	const nanovdb::Vec3f u_pZ = velocitySampler(c + nanovdb::Coord(0, 0, 1));
	const nanovdb::Vec3f u_mZ = velocitySampler(c - nanovdb::Coord(0, 0, 1));

	const float omega_x = ((u_pY[2] - u_mY[2]) - (u_pZ[1] - u_mZ[1])) * factor;
	const float omega_y = ((u_pZ[0] - u_mZ[0]) - (u_pX[2] - u_mX[2])) * factor;
	const float omega_z = ((u_pX[1] - u_mX[1]) - (u_pY[0] - u_mY[0])) * factor;


	// X-direction neighbors
	const float vortMag_pX = computeVorticityMag(velocitySampler, c + nanovdb::Coord(factorScale, 0, 0), factor);
	const float vortMag_mX = computeVorticityMag(velocitySampler, c - nanovdb::Coord(factorScale, 0, 0), factor);

	// Y-direction neighbors
	const float vortMag_pY = computeVorticityMag(velocitySampler, c + nanovdb::Coord(0, factorScale, 0), factor);
	const float vortMag_mY = computeVorticityMag(velocitySampler, c - nanovdb::Coord(0, factorScale, 0), factor);

	// Z-direction neighbors
	const float vortMag_pZ = computeVorticityMag(velocitySampler, c + nanovdb::Coord(0, 0, factorScale), factor);
	const float vortMag_mZ = computeVorticityMag(velocitySampler, c - nanovdb::Coord(0, 0, factorScale), factor);

	// Compute gradient with safe differences
	const float grad_x = (vortMag_pX - vortMag_mX) * 0.5f * inv_dx;
	const float grad_y = (vortMag_pY - vortMag_mY) * 0.5f * inv_dx;
	const float grad_z = (vortMag_pZ - vortMag_mZ) * 0.5f * inv_dx;

	// Normalize gradient vector
	const float gradLen = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z) + 1e-5f;
	const float Nx = grad_x / gradLen;
	const float Ny = grad_y / gradLen;
	const float Nz = grad_z / gradLen;

	// Compute confinement force using cross product
	outForce[tid] = velocityData[tid] + nanovdb::Vec3f{confinementScale * (Ny * omega_z - Nz * omega_y),
	                                                   confinementScale * (Nz * omega_x - Nx * omega_z),
	                                                   confinementScale * (Nx * omega_y - Ny * omega_x)} *
	                                        dt;
}