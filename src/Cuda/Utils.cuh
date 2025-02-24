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

template <typename T>
__device__ __forceinline__ T computeError(const T& value, const T& value_backward) {
	return static_cast<T>(0.5f) * (value - value_backward);
}

__device__ __forceinline__ float absValue(const float x) { return fabsf(x); }

__device__ __forceinline__ nanovdb::Vec3f absValue(const nanovdb::Vec3f& v) {
	return nanovdb::Vec3f(fabsf(v[0]), fabsf(v[1]), fabsf(v[2]));
}

template <typename T>
__device__ __forceinline__ T computeMaxCorrection(const T& value_forward, const T& value) {
	return static_cast<T>(0.5f) * absValue(value_forward - value);
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

__device__ __forceinline__ nanovdb::Vec3f enforceNonNegative(const nanovdb::Vec3f& v) { return v; }


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

inline __device__ nanovdb::Vec3f MACToFaceCentered(
    const decltype(nanovdb::math::createSampler<1>(
        std::declval<const decltype(std::declval<nanovdb::Vec3fGrid>().tree().getAccessor())>()))& velSampler,
    const nanovdb::Vec3f& pos) {
	const float up = velSampler(pos + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float vp = velSampler(pos + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float wp = velSampler(pos + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	const float um = velSampler(pos - nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float vm = velSampler(pos - nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float wm = velSampler(pos - nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	return {(up + um) / 2.0f, (vp + vm) / 2.0f, (wp + wm) / 2.0f};
}