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

	const nanovdb::Vec3f forward_pos = rk3_integrate(velocitySampler, pos, -scaled_dt);
	const float value_forward = dataSampler(forward_pos);

	const nanovdb::Vec3f back_pos = rk3_integrate(velocitySampler, forward_pos, scaled_dt);
	const float value_backward = dataSampler(back_pos);

	const float error = computeError(original, value_backward);
	float value_corrected = __fadd_rn(value_forward, error);

	const float max_correction = computeMaxCorrection(value_forward, original);
	value_corrected = fminf(value_forward + max_correction, fmaxf(value_forward - max_correction, value_corrected));
	value_corrected = fmaxf(0.0f, value_corrected);

	outData[idx] = value_corrected;
}

inline __global__ void advect_vector(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ coords,
                                     const nanovdb::Vec3f* __restrict__ velocityData, nanovdb::Vec3f* __restrict__ outVelocity,
                                     const size_t totalVoxels, const float dt, const float voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float inv_voxelSize = 1.0f / voxelSize;
	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f orig_vel = velocityData[idx];
	const nanovdb::Vec3f pos = coord.asVec3s();

	const nanovdb::Vec3f forward_pos = rk3_integrate(velocitySampler, pos, -scaled_dt);
	const nanovdb::Vec3f value_forward = velocitySampler(forward_pos);

	const nanovdb::Vec3f back_pos = rk3_integrate(velocitySampler, forward_pos, scaled_dt);
	const nanovdb::Vec3f value_backward = velocitySampler(back_pos);

	const nanovdb::Vec3f error = computeError(orig_vel, value_backward);
	nanovdb::Vec3f value_corrected = value_forward + error;
	const nanovdb::Vec3f max_correction = computeMaxCorrection(value_forward, orig_vel);

	value_corrected = clampValue(value_corrected, value_forward - max_correction, value_forward + max_correction);

	outVelocity[idx] = value_corrected;
}

inline __global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coord,
                                  const nanovdb::Vec3f* __restrict__ velocityData, float* __restrict__ outDivergence, const float dx,
                                  const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const nanovdb::Coord coord = d_coord[tid];

	if (!velocitySampler.isDataActive(coord)) {
		return;
	}

	const nanovdb::Vec3f c = coord.asVec3s();

	const float xp = velocitySampler(c + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float xm = velocitySampler(c - nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float yp = velocitySampler(c + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float ym = velocitySampler(c - nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float zp = velocitySampler(c + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	const float zm = velocitySampler(c - nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	const float dixX = (xp - xm) / dx;
	const float dixY = (yp - ym) / dx;
	const float dixZ = (zp - zm) / dx;

	outDivergence[tid] = dixX + dixY + dixZ;
}

inline __global__ void redBlackGaussSeidelUpdate(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
                                                 const nanovdb::Coord* __restrict__ d_coord, const float* __restrict__ divergence,
                                                 float* __restrict__ pressure, const float dx, const size_t totalVoxels, const int color,
                                                 const float omega) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Coord c = d_coord[tid];
	const int i = c.x(), j = c.y(), k = c.z();

	// Skip if wrong color (reduces divergent branching)
	if ((i + j + k & 1) != color) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto divSampler = IndexSampler<float, 1>(idxSampler, divergence);
	const auto pSampler = IndexSampler<float, 1>(idxSampler, pressure);

	// Pre-compute common factors
	const float dx2 = dx * dx;
	const float inv6 = 1.0f / 6.0f;

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


inline __global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
                                                const nanovdb::Coord* __restrict__ d_coords, const size_t totalVoxels,
                                                const nanovdb::Vec3f* __restrict__ velocity,  // velocity at faces
                                                const float* __restrict__ pressure,           // pressure at cell centers
                                                nanovdb::Vec3f* __restrict__ out, float voxelSize) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	// Accessors / Samplers
	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto pressureSampler = IndexSampler<float, 1>(idxSampler, pressure);
	const auto velSampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocity);


	const float dx = voxelSize;

	// The cell center coordinate
	const nanovdb::Vec3f c = d_coords[tid].asVec3s();
	const nanovdb::Vec3f vel = sampleMACVelocity_idx(velSampler, c);
	nanovdb::Vec3f v;

	// For u component: Sample velocity at (i+1/2,j,k) relative to cell center
	{
		// For x-component, we're already at the face center
		const float p_left = pressureSampler(c);                                   // p(i,j,k)
		const float p_right = pressureSampler(c + nanovdb::Vec3f(1, 0.0f, 0.0f));  // p(i+1,j,k)
		v[0] = vel[0] - (p_right - p_left) / dx;
	}

	// For v component: Sample velocity at (i,j+1/2,k) relative to cell center
	{
		const float p_bottom = pressureSampler(c);                               // p(i,j,k)
		const float p_top = pressureSampler(c + nanovdb::Vec3f(0.0f, 1, 0.0f));  // p(i,j+1,k)
		v[1] = vel[1] - (p_top - p_bottom) / dx;
	}

	// For w component: Sample velocity at (i,j,k+1/2) relative to cell center
	{
		const float p_back = pressureSampler(c);                                   // p(i,j,k)
		const float p_front = pressureSampler(c + nanovdb::Vec3f(0.0f, 0.0f, 1));  // p(i,j,k+1)
		v[2] = vel[2] - (p_front - p_back) / dx;
	}

	out[tid] = v;
}

inline __global__ void vel_y_density(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ coords,
                                     const nanovdb::Vec3f* __restrict__ velocityData, const float* __restrict__ densityData,
                                     nanovdb::Vec3f* __restrict__ outVel, const size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto densitySampler = IndexSampler<float, 1>(idxSampler, densityData);

	const nanovdb::Coord coord = coords[idx];

	const float den_at_idx = densitySampler(coord);
	outVel[idx] = velocitySampler(coord) + nanovdb::Vec3f(0, den_at_idx, 0);
}