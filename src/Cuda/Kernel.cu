#pragma once

#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"


__global__ void advect_scalar(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              const float* __restrict__ inData, float* __restrict__ outData, const size_t totalVoxels, const float dt,
                              const float inv_voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto dataSampler = IndexSampler<float, 1>(idxSampler, inData);

	// The cell coordinate
	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f posCell = coord.asVec3s();
	const float phiOrig = inData[idx];

	// MAC Velocity for backtrace
	const nanovdb::Vec3f velCenter = MACToFaceCentered(velocitySampler, coord);

	// ------------------------------------------------------------------------
	// Forward pass (semi-Lagrangian backtrace)
	// x_forward = x - vel*dt
	const nanovdb::Vec3f backPos = posCell - velCenter * scaled_dt;
	const float phiForward = dataSampler(backPos);

	// ------------------------------------------------------------------------
	// Backward pass
	// x_backward = x_forward + u(x_forward)*dt
	// Then compare that to the original value
	const nanovdb::Vec3f velF = velocitySampler(backPos);
	const nanovdb::Vec3f fwdPos2 = backPos + velF * scaled_dt;
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
                              nanovdb::Vec3f* __restrict__ outVelocity, const size_t totalVoxels, const float dt,
                              const float inv_voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f pos = coord.asVec3s();

	// Original velocity at the cell or face
	const nanovdb::Vec3f velOrig = MACToFaceCentered(velocitySampler, coord);

	// Forward pass (backtrace)
	const nanovdb::Vec3f backPos = pos - velOrig * scaled_dt;
	const nanovdb::Vec3f velForward = velocitySampler(backPos);

	// Backward check
	const nanovdb::Vec3f fwdPos2 = backPos + velForward * scaled_dt;
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

	outVelocity[idx] = velCorr;
}

__global__ void divergence_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Vec3f* velocityData,
                               float* outDivergence, const float inv_dx, const int numLeaves) {
	// Block dimensions matching leaf size
	constexpr int BLOCK_SIZE = 8;
	__shared__ nanovdb::Vec3f s_vel[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	const auto leafptr = domainGrid->tree().getFirstNode<0>();

	const int leafIdx = blockIdx.x;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;

	if (leafIdx >= numLeaves) return;

	const nanovdb::Coord origin = leafptr[leafIdx].origin();

	const auto sampler = IndexOffsetSampler<0>(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(sampler, velocityData);

	for (int idx = tidz * BLOCK_SIZE * BLOCK_SIZE + tidy * BLOCK_SIZE + tidx; idx < (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2);
	     idx += BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) {
		const int dz = idx / ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2));
		const int dy = (idx % ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))) / (BLOCK_SIZE + 2);
		const int dx = idx % (BLOCK_SIZE + 2);

		const nanovdb::Coord coord(origin.x() + dx - 1, origin.y() + dy - 1, origin.z() + dz - 1);

		s_vel[dz][dy][dx] = velocitySampler(coord);
	}

	__syncthreads();

	const int lx = tidx + 1;
	const int ly = tidy + 1;
	const int lz = tidz + 1;

	// Load current cell's velocity
	const nanovdb::Vec3f current = s_vel[lz][ly][lx];

	// Average neighboring velocities for each component
	const float xp = 0.5f * (current[0] + s_vel[lz][ly][lx + 1][0]);
	const float xm = 0.5f * (current[0] + s_vel[lz][ly][lx - 1][0]);

	const float yp = 0.5f * (current[1] + s_vel[lz][ly + 1][lx][1]);
	const float ym = 0.5f * (current[1] + s_vel[lz][ly - 1][lx][1]);

	const float zp = 0.5f * (current[2] + s_vel[lz + 1][ly][lx][2]);
	const float zm = 0.5f * (current[2] + s_vel[lz - 1][ly][lx][2]);

	const float divergence = (xp - xm + yp - ym + zp - zm) * inv_dx;

	outDivergence[sampler.offset(origin + nanovdb::Coord(tidx, tidy, tidz))] = divergence;
}


__global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                           const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                           float* __restrict__ outDivergence, const float inv_dx, const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Vec3f center = velocityData[tid];

	const nanovdb::Coord c = d_coord[tid];
	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocityData);

	const float xp = (center + velocitySampler(c + nanovdb::Coord(1, 0, 0)))[0] * 0.5f;
	const float xm = (center + velocitySampler(c - nanovdb::Coord(1, 0, 0)))[0] * 0.5f;
	const float yp = (center + velocitySampler(c + nanovdb::Coord(0, 1, 0)))[1] * 0.5f;
	const float ym = (center + velocitySampler(c - nanovdb::Coord(0, 1, 0)))[1] * 0.5f;
	const float zp = (center + velocitySampler(c + nanovdb::Coord(0, 0, 1)))[2] * 0.5f;
	const float zm = (center + velocitySampler(c - nanovdb::Coord(0, 0, 1)))[2] * 0.5f;

	outDivergence[tid] = (xp - xm + yp - ym + zp - zm) * inv_dx;
}

__global__ void redBlackGaussSeidelUpdate_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                              const float* __restrict__ d_divergence, float* __restrict__ d_pressure, float voxelSize,
                                              int numLeaves, int color, float omega) {
	constexpr int BLOCK_SIZE = 8;
	__shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];


	// Which leaf block am I?
	const int leafIdx = blockIdx.x;
	if (leafIdx >= numLeaves) return;

	const auto& leaf = domainGrid->tree().getFirstNode<0>()[leafIdx];
	const nanovdb::Coord origin = leaf.origin();

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto divSampler = IndexSampler<float, 0>(idxSampler, d_divergence);
	const auto pSampler = IndexSampler<float, 0>(idxSampler, d_pressure);

	// Thread indices
	const int tidz = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.z;


	for (int idx = tidz * BLOCK_SIZE * BLOCK_SIZE + tidy * BLOCK_SIZE + tidx; idx < (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2);
	     idx += BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) {
		const int dz = idx / ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2));
		const int dy = (idx % ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))) / (BLOCK_SIZE + 2);
		const int dx = idx % (BLOCK_SIZE + 2);

		const nanovdb::Coord coord(origin.x() + dx - 1, origin.y() + dy - 1, origin.z() + dz - 1);

		s_pressure[dz][dy][dx] = pSampler(coord);
	}


	if (((tidx + tidy + tidx) & 1) != color) return;


	__syncthreads();

	if (tidx < BLOCK_SIZE && tidy < BLOCK_SIZE && tidz < BLOCK_SIZE) {
		const int lx = tidx + 1;
		const int ly = tidy + 1;
		const int lz = tidz + 1;

		// Pre-compute common factors
		const float dx2 = voxelSize * voxelSize;
		constexpr float inv6 = 0.166666667;

		const nanovdb::Coord c = origin + nanovdb::Coord(tidx, tidy, tidz);
		const float divVal = divSampler(c);
		const float pOld = pSampler(c);

		const float pxp1 = s_pressure[lz][ly][tidx + 1];
		const float pxm1 = s_pressure[lz][ly][tidx];
		const float pyp1 = s_pressure[lz][tidy + 1][lx];
		const float pym1 = s_pressure[lz][tidy][lx];
		const float pzp1 = s_pressure[tidz + 1][ly][lx];
		const float pzm1 = s_pressure[tidz][ly][lx];

		const float pGS = ((pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1) - divVal * dx2) * inv6;
		d_pressure[idxSampler.offset(c)] = pOld + omega * (pGS - pOld);
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

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
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
	__shared__ nanovdb::Vec3f s_vel[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	__shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	const int leafIdx = blockIdx.x;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;

	if (leafIdx >= numLeaves) return;

	const auto& leaf = domainGrid->tree().getFirstNode<0>()[leafIdx];
	const nanovdb::Coord origin = leaf.origin();

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocity);
	const auto pressureSampler = IndexSampler<float, 0>(idxSampler, pressure);

	for (int idx = tidz * BLOCK_SIZE * BLOCK_SIZE + tidy * BLOCK_SIZE + tidx; idx < (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2);
	     idx += BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) {
		const int dz = idx / ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2));
		const int dy = (idx % ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))) / (BLOCK_SIZE + 2);
		const int dx = idx % (BLOCK_SIZE + 2);

		const nanovdb::Coord coord(origin.x() + dx - 1, origin.y() + dy - 1, origin.z() + dz - 1);

		s_vel[dz][dy][dx] = velocitySampler(coord);
		s_pressure[dz][dy][dx] = pressureSampler(coord);
	}

	__syncthreads();

	if (tidx < BLOCK_SIZE && tidy < BLOCK_SIZE && tidz < BLOCK_SIZE) {
		const int lx = tidx + 1;
		const int ly = tidy + 1;
		const int lz = tidz + 1;

		// Load current cell's
		const nanovdb::Vec3f v_c = s_vel[lz][ly][lx];
		const float p_c = s_pressure[lz][ly][lx];

		nanovdb::Vec3f v;

		const float u = (v_c + s_vel[lz][ly][lx + 1])[0] * 0.5f;
		const float p_right = s_pressure[lz][ly][lx + 1];
		v[0] = u - (p_right - p_c) * inv_voxelSize;

		const float v_comp = (v_c + s_vel[lz][ly + 1][lx])[1] * 0.5f;
		const float p_top = s_pressure[lz][ly + 1][lx];
		v[1] = v_comp - (p_top - p_c) * inv_voxelSize;

		const float w = (v_c + s_vel[lz + 1][ly][lx])[2] * 0.5f;
		const float p_front = s_pressure[lz + 1][ly][lx];
		v[2] = w - (p_front - p_c) * inv_voxelSize;

		out[idxSampler.offset(origin + nanovdb::Coord(tidx, tidy, tidz))] = v;
	}
}


__global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                         const nanovdb::Coord* __restrict__ d_coords, const size_t totalVoxels,
                                         const nanovdb::Vec3f* __restrict__ velocity, const float* __restrict__ pressure,
                                         nanovdb::Vec3f* __restrict__ out, const float inv_voxelSize) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	// Accessors / Samplers
	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	// The cell center coordinate
	const nanovdb::Coord& c = d_coords[tid];
	const nanovdb::Vec3f& vel_c = velocity[tid];

	const auto pressureSampler = IndexSampler<float, 0>(idxSampler, pressure);
	const auto velSampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocity);

	nanovdb::Vec3f v;

	// For u-component at (i+1/2,j,k)
	// Use both neighbors for velocity averaging, matching the divergence calculation
	const nanovdb::Vec3f vel_xp = velSampler(c + nanovdb::Coord(1, 0, 0));

	// Use central differencing for pressure
	const float p_right = pressureSampler(c + nanovdb::Coord(1, 0, 0));  // p(i+1,j,k)
	const float p_left = pressureSampler(c - nanovdb::Coord(1, 0, 0));   // p(i-1,j,k)

	// Average velocity from neighbors (central)
	const float u = (vel_c[0] + vel_xp[0]) * 0.5f;

	// Apply symmetric pressure gradient
	v[0] = u - (p_right - p_left) * 0.5f * inv_voxelSize;

	// For v-component at (i,j+1/2,k)
	const nanovdb::Vec3f vel_yp = velSampler(c + nanovdb::Coord(0, 1, 0));

	const float p_top = pressureSampler(c + nanovdb::Coord(0, 1, 0));     // p(i,j+1,k)
	const float p_bottom = pressureSampler(c - nanovdb::Coord(0, 1, 0));  // p(i,j-1,k)

	// Average velocity from neighbors (central)
	const float v_comp = (vel_c[1] + vel_yp[1]) * 0.5f;

	// Apply symmetric pressure gradient
	v[1] = v_comp - (p_top - p_bottom) * 0.5f * inv_voxelSize;

	// For w-component at (i,j,k+1/2)
	const nanovdb::Vec3f vel_zp = velSampler(c + nanovdb::Coord(0, 0, 1));

	const float p_front = pressureSampler(c + nanovdb::Coord(0, 0, 1));  // p(i,j,k+1)
	const float p_back = pressureSampler(c - nanovdb::Coord(0, 0, 1));   // p(i,j,k-1)

	// Average velocity from neighbors (central)
	const float w = (vel_c[2] + vel_zp[2]) * 0.5f;

	// Apply symmetric pressure gradient
	v[2] = w - (p_front - p_back) * 0.5f * inv_voxelSize;

	out[tid] = v;
}


__global__ void temperature_buoyancy(const nanovdb::Vec3f* velocityData, const float* tempData, nanovdb::Vec3f* outVel, const float dt,
                                     float ambient_temp, float buoyancy_strength, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	// Get current temperature and velocity
	const float temp = tempData[idx];
	const nanovdb::Vec3f vel = velocityData[idx];

	// Calculate temperature difference from ambient
	const float tempDiff = temp - ambient_temp;

	// Add buoyancy force proportional to temperature difference
	// High temperatures cause upward motion (positive Y in most simulation setups)
	const nanovdb::Vec3f buoyancyForce(0.0f, max(0.0f, tempDiff * buoyancy_strength), 0.0f);

	// Update velocity with buoyancy
	outVel[idx] = vel + buoyancyForce * dt;
}


__global__ void combustion(const float* fuelData, const float* tempData, float* outFuel, float* outTemp, const float dt,
                           float ignition_temp, float combustion_rate, float heat_release, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	// Get current state
	const float fuel = fuelData[idx];
	const float temp = tempData[idx];

	// Initialize output values with input values
	float newFuel = fuel;
	float newTemp = temp;

	// Only process combustion if we have fuel and temperature is above ignition point
	if (fuel > 0.0f && temp >= ignition_temp) {
		// Calculate how much fuel is burned this step
		const float fuelBurned = min(fuel, combustion_rate * dt);

		// Update fuel level by reducing it
		newFuel -= fuelBurned;

		// Update temperature based on heat release from combustion
		newTemp += fuelBurned * heat_release;
	}

	// Store results
	outFuel[idx] = newFuel;
	outTemp[idx] = newTemp;
}


__global__ void diffusion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                          const float* tempData, const float* fuelData, float* outTemp, float* outFuel, const float dt, float temp_diff,
                          float fuel_diff, float ambient_temp, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	// Get current voxel coordinate
	const auto sampler = IndexOffsetSampler<0>(*domainGrid);

	// Get current temperature and fuel values
	const float centerTemp = tempData[idx];
	const float centerFuel = fuelData[idx];

	// Perform a simple 6-neighbor stencil diffusion
	float tempLaplacian = 0.0f;
	float fuelLaplacian = 0.0f;
	int neighbors = 0;

	// Check each of the 6 direct neighbors
	nanovdb::Coord offsets[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

	for (auto offset : offsets) {
		const nanovdb::Coord neighborCoord = d_coords[idx] + offset;

		// Skip if neighbor is outside the domain
		if (!sampler.isActive(neighborCoord)) continue;

		// Get the neighbor's index and values
		const uint64_t neighborIdx = sampler.offset(neighborCoord);
		const float neighborTemp = tempData[neighborIdx];
		const float neighborFuel = fuelData[neighborIdx];

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

	// Apply cooling - temperature gradually returns to ambient
	outTemp[idx] = outTemp[idx] + (ambient_temp - outTemp[idx]) * dt * 0.1f;
}