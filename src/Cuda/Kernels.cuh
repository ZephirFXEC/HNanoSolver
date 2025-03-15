#pragma once

#include "../Utils/Stencils.hpp"
#include "Utils.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

inline __global__ void advect_scalar(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                     const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                                     const float* __restrict__ inData, float* __restrict__ outData, const size_t totalVoxels,
                                     const float dt, const float inv_voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const nanovdb::Coord coord = coords[idx];
	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto dataSampler = IndexSampler<float, 1>(idxSampler, inData);

	const nanovdb::Vec3f pos = coord.asVec3s();
	const float original = inData[idx];

	// Forward advection (backward in time)
	const nanovdb::Vec3f forward_pos = rk4_integrate(velocitySampler, pos, -scaled_dt);
	const float value_forward = dataSampler(forward_pos);

	// Backward advection (forward in time) from the forward position
	const nanovdb::Vec3f back_pos = rk4_integrate(velocitySampler, forward_pos, scaled_dt);
	const float value_backward = dataSampler(back_pos);

	// Compute error and apply correction
	const float error = computeError(original, value_backward);
	float value_corrected = __fadd_rn(value_forward, error);

	// Limit correction magnitude
	const float max_correction = computeMaxCorrection(value_forward, original);
	value_corrected = fminf(value_forward + max_correction, fmaxf(value_forward - max_correction, value_corrected));
	value_corrected = fmaxf(0.0f, value_corrected);

	outData[idx] = value_corrected;
}

inline __global__ void advect_vector(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                     const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                                     nanovdb::Vec3f* __restrict__ outVelocity, const size_t totalVoxels, const float dt,
                                     const float inv_voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f orig_vel = velocityData[idx];
	const nanovdb::Vec3f pos = coord.asVec3s();

	// Forward advection (backward in time)
	const nanovdb::Vec3f forward_pos = rk4_integrate(velocitySampler, pos, -scaled_dt);
	const nanovdb::Vec3f value_forward = velocitySampler(forward_pos);

	// Backward advection (forward in time) from the forward position
	const nanovdb::Vec3f back_pos = rk4_integrate(velocitySampler, forward_pos, scaled_dt);
	const nanovdb::Vec3f value_backward = velocitySampler(back_pos);

	// Compute error and apply correction
	const nanovdb::Vec3f error = computeError(orig_vel, value_backward);
	nanovdb::Vec3f value_corrected = value_forward + error;
	const nanovdb::Vec3f max_correction = computeMaxCorrection(value_forward, orig_vel);

	value_corrected = clampValue(value_corrected, value_forward - max_correction, value_forward + max_correction);

	outVelocity[idx] = value_corrected;
}

inline __global__ void divergence_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Vec3f* velocityData,
                                      float* outDivergence, const float inv_dx, const int numLeaves) {
	// Block dimensions matching leaf size
	constexpr int BLOCK_SIZE = 8;
	__shared__ nanovdb::Vec3f s_vel[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	const int leafIdx = blockIdx.x;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;

	if (leafIdx >= numLeaves) return;

	const auto& leaf = domainGrid->tree().getFirstNode<0>()[leafIdx];
	const nanovdb::Coord origin = leaf.origin();

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto accessor = domainGrid->getAccessor();

	for (int idx = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	     idx < (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2); idx += blockDim.x * blockDim.y * blockDim.z) {
		const int dz = idx / ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2));
		const int dy = (idx % ((BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))) / (BLOCK_SIZE + 2);
		const int dx = idx % (BLOCK_SIZE + 2);

		const nanovdb::Coord coord = origin + nanovdb::Coord(dx - 1, dy - 1, dz - 1);

		s_vel[dz][dy][dx] = accessor.isActive(coord) ? velocityData[accessor.getValue(coord)] : nanovdb::Vec3f(0.0f);
	}

	__syncthreads();

	if (tidx < BLOCK_SIZE && tidy < BLOCK_SIZE && tidz < BLOCK_SIZE) {
		// Halo offset indices
		const int lx = tidx + 1;
		const int ly = tidy + 1;
		const int lz = tidz + 1;

		// Stencil computation using shared memory
		const float xp = s_vel[lz][ly][lx + 1][0];
		const float xm = s_vel[lz][ly][lx - 1][0];
		const float yp = s_vel[lz][ly + 1][lx][1];
		const float ym = s_vel[lz][ly - 1][lx][1];
		const float zp = s_vel[lz + 1][ly][lx][2];
		const float zm = s_vel[lz - 1][ly][lx][2];

		const float divergence = (xp - xm + yp - ym + zp - zm) * inv_dx;

		// Boundary condition (origin check)
		const nanovdb::Coord globalCoord = origin + nanovdb::Coord(tidx, tidy, tidz);
		const bool isBoundary = (globalCoord.x() == 0) || (globalCoord.y() == 0) || (globalCoord.z() == 0);

		// Store result
		outDivergence[leaf.getValue(origin + nanovdb::Coord(tidx, tidy, tidz))] = isBoundary ? 0.0f : divergence;
	}
}


inline __global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                  const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                                  float* __restrict__ outDivergence, const float inv_dx, const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const nanovdb::Coord coord = d_coord[tid];

	const nanovdb::Vec3f c = coord.asVec3s();

	const float xp = velocitySampler(c + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float xm = velocitySampler(c - nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float yp = velocitySampler(c + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float ym = velocitySampler(c - nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float zp = velocitySampler(c + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	const float zm = velocitySampler(c - nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	const float dixX = (xp - xm) * inv_dx;
	const float dixY = (yp - ym) * inv_dx;
	const float dixZ = (zp - zm) * inv_dx;

	outDivergence[tid] = dixX + dixY + dixZ;
}

inline __global__ void redBlackGaussSeidelUpdate(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                                 const nanovdb::Coord* __restrict__ d_coord, const float* __restrict__ divergence,
                                                 float* __restrict__ pressure, const float dx, const size_t totalVoxels, const int color,
                                                 const float omega) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Coord c = d_coord[tid];
	const int i = c.x(), j = c.y(), k = c.z();

	// Skip if wrong color (reduces divergent branching)
	if (((i + j + k) & 1) != color) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto divSampler = IndexSampler<float, 0>(idxSampler, divergence);
	const auto pSampler = IndexSampler<float, 0>(idxSampler, pressure);

	// Pre-compute common factors
	const float dx2 = dx * dx;
	constexpr float inv6 = 0.166666667;

	// Gather neighbors using vectorized loads where possible
	const float pxp1 = pSampler(nanovdb::Coord(i + 1, j, k));
	const float pxm1 = pSampler(nanovdb::Coord(i - 1, j, k));
	const float pyp1 = pSampler(nanovdb::Coord(i, j + 1, k));
	const float pym1 = pSampler(nanovdb::Coord(i, j - 1, k));
	const float pzp1 = pSampler(nanovdb::Coord(i, j, k + 1));
	const float pzm1 = pSampler(nanovdb::Coord(i, j, k - 1));

	const float divVal = divSampler(c);
	const float pOld = pSampler(c);

	// Optimize arithmetic operations
	const float pGS = ((pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1) - divVal * dx2) * inv6;
	pressure[tid] = pOld + omega * (pGS - pOld);
}


inline __global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                                const nanovdb::Coord* __restrict__ d_coords, const size_t totalVoxels,
                                                const nanovdb::Vec3f* __restrict__ velocity, const float* __restrict__ pressure,
                                                nanovdb::Vec3f* __restrict__ out, const float inv_voxelSize) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	// Accessors / Samplers
	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto pressureSampler = IndexSampler<float, 0>(idxSampler, pressure);
	const auto velSampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocity);

	// The cell center coordinate
	const nanovdb::Coord c = d_coords[tid];
	const nanovdb::Vec3f pos = c.asVec3s();
	nanovdb::Vec3f v;

	// For u-component at (i+1/2,j,k)
	// The pressure gradient at u-face is between p(i,j,k) and p(i+1,j,k)
	const float u = velSampler(pos + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float p_c = pressureSampler(c);                                // p(i,j,k)
	const float p_right = pressureSampler(c + nanovdb::Coord(1, 0, 0));  // p(i+1,j,k)
	v[0] = u - (p_right - p_c) * inv_voxelSize;

	// For v-component at (i,j+1/2,k)
	// The pressure gradient at v-face is between p(i,j,k) and p(i,j+1,k)
	const float v_comp = velSampler(pos + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float p_top = pressureSampler(c + nanovdb::Coord(0, 1, 0));  // p(i,j+1,k)
	v[1] = v_comp - (p_top - p_c) * inv_voxelSize;

	// For w-component at (i,j,k+1/2)
	// The pressure gradient at w-face is between p(i,j,k) and p(i,j,k+1)
	const float w = velSampler(pos + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	const float p_front = pressureSampler(c + nanovdb::Coord(0, 0, 1));  // p(i,j,k+1)
	v[2] = w - (p_front - p_c) * inv_voxelSize;

	out[tid] = v;
}

inline __global__ void vel_y_density(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                     const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                                     const float* __restrict__ densityData, nanovdb::Vec3f* __restrict__ outVel, const size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto densitySampler = IndexSampler<float, 1>(idxSampler, densityData);

	const nanovdb::Coord coord = coords[idx];

	const float den_at_idx = densitySampler(coord);
	outVel[idx] = velocitySampler(coord) + nanovdb::Vec3f(0, den_at_idx, 0);
}