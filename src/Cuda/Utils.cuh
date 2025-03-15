#pragma once

#include "../Utils/Stencils.hpp"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/SampleFromVoxels.h"

template <typename T>
__device__ T lerp(T v0, T v1, T t) {
	return fma(t, v1, fma(-t, v0, v0));
}

inline size_t blocksPerGrid(const size_t numItems, const size_t threadsPerBlock) {
	NANOVDB_ASSERT(numItems > 0 && threadsPerBlock >= 32 && threadsPerBlock % 32 == 0);
	return (numItems + threadsPerBlock - 1) / threadsPerBlock;
}

__device__ __forceinline__ float computeError(const float original, const float backward) {
	return __fmul_rn(0.5f, __fsub_rn(original, backward));
}

__device__ __forceinline__ nanovdb::Vec3f computeError(const nanovdb::Vec3f& original, const nanovdb::Vec3f& backward) {
	return nanovdb::Vec3f(computeError(original[0], backward[0]), computeError(original[1], backward[1]),
	                      computeError(original[2], backward[2]));
}

__device__ __forceinline__ float absValue(const float x) { return fabsf(x); }

__device__ __forceinline__ nanovdb::Vec3f absValue(const nanovdb::Vec3f& v) {
	return nanovdb::Vec3f(fabsf(v[0]), fabsf(v[1]), fabsf(v[2]));
}

__device__ __forceinline__ float computeMaxCorrection(const float forward, const float original) {
	return __fmul_rn(0.5f, fabsf(__fsub_rn(forward, original)));
}
__device__ __forceinline__ nanovdb::Vec3f computeMaxCorrection(const nanovdb::Vec3f& forward, const nanovdb::Vec3f& original) {
	return nanovdb::Vec3f(computeMaxCorrection(forward[0], original[0]), computeMaxCorrection(forward[1], original[1]),
	                      computeMaxCorrection(forward[2], original[2]));
}


__device__ __forceinline__ float clampValue(const float x, const float minVal, const float maxVal) {
	return fminf(fmaxf(x, minVal), maxVal);
}

__device__ __forceinline__ nanovdb::Vec3f clampValue(const nanovdb::Vec3f& x, const nanovdb::Vec3f& minVal, const nanovdb::Vec3f& maxVal) {
	return nanovdb::Vec3f(fminf(fmaxf(x[0], minVal[0]), maxVal[0]), fminf(fmaxf(x[1], minVal[1]), maxVal[1]),
	                      fminf(fmaxf(x[2], minVal[2]), maxVal[2]));
}

__device__ __forceinline__ float lerp(const float a, const float b, const float t) { return a + t * (b - a); }

__device__ __forceinline__ nanovdb::Vec3f lerp(const nanovdb::Vec3f& a, const nanovdb::Vec3f& b, const float t) { return a + (b - a) * t; }

__device__ __forceinline__ float enforceNonNegative(const float x) { return fmaxf(0.0f, x); }


inline __device__ nanovdb::Vec3f sampleMACVelocity(
    const decltype(nanovdb::math::createSampler<1>(
        std::declval<const decltype(std::declval<nanovdb::Vec3fGrid>().tree().getAccessor())>()))& velSampler,
    const nanovdb::Vec3f& pos) {
	const float u = velSampler(pos + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float v = velSampler(pos + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float w = velSampler(pos + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	return {u, v, w};
}

inline __device__ nanovdb::Vec3f sampleMACVelocity_idx(const IndexSampler<nanovdb::Vec3f, 1>& velSampler, const nanovdb::Vec3f& pos) {
	const float u = velSampler(pos + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float v = velSampler(pos + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float w = velSampler(pos + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	return {u, v, w};
}


inline __device__ nanovdb::Vec3f MACToFaceCentered_idx(const IndexSampler<nanovdb::Vec3f, 1>& velSampler, const nanovdb::Vec3f& pos) {
	const float up = velSampler(pos + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float vp = velSampler(pos + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float wp = velSampler(pos + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	const float um = velSampler(pos - nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float vm = velSampler(pos - nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float wm = velSampler(pos - nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	return {(up + um) / 2.0f, (vp + vm) / 2.0f, (wp + wm) / 2.0f};
}


template <typename VelocitySampler>
__device__ nanovdb::Vec3f rk4_integrate(const VelocitySampler& sampler, nanovdb::Vec3f start_pos, float h) {
	const nanovdb::Vec3f k1 = sampler(start_pos) * h;
	const nanovdb::Vec3f k2 = sampler(start_pos + 0.5f * k1) * h;
	const nanovdb::Vec3f k3 = sampler(start_pos + 0.5f * k2) * h;
	const nanovdb::Vec3f k4 = sampler(start_pos + k3) * h;

	nanovdb::Vec3f result = start_pos;
	result[0] = __fmaf_rn(0.16667f, k1[0] + k4[0] + 2.0f * (k2[0] + k3[0]), start_pos[0]);
	result[1] = __fmaf_rn(0.16667f, k1[1] + k4[1] + 2.0f * (k2[1] + k3[1]), start_pos[1]);
	result[2] = __fmaf_rn(0.16667f, k1[2] + k4[2] + 2.0f * (k2[2] + k3[2]), start_pos[2]);
	return result;
}

template <typename VelocitySampler>
__device__ nanovdb::Vec3f rk3_integrate(const VelocitySampler& sampler, nanovdb::Vec3f start_pos, float h) {
	const nanovdb::Vec3f k1 = sampler(start_pos) * h;
	const nanovdb::Vec3f k2 = sampler(start_pos + 0.5f * k1) * h;
	const nanovdb::Vec3f k3 = sampler(start_pos - k1 + 2.0f * k2) * h;

	nanovdb::Vec3f result = start_pos;
	result[0] = __fmaf_rn(0.33333f, k1[0] + 3.0f * k2[0] + k3[0], start_pos[0]);
	result[1] = __fmaf_rn(0.33333f, k1[1] + 3.0f * k2[1] + k3[1], start_pos[1]);
	result[2] = __fmaf_rn(0.33333f, k1[2] + 3.0f * k2[2] + k3[2], start_pos[2]);
	return result;
}


class ScopedTimerGPU {
   public:
	explicit ScopedTimerGPU(std::string name) : name_(std::move(name)) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}
	~ScopedTimerGPU() {
		float elapsed;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);

		printf("%s Time: %f ms\n", name_.c_str(), elapsed);
	}

   private:
	const std::string name_{};
	cudaEvent_t start{};
	cudaEvent_t stop{};
};