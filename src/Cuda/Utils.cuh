#pragma once

#include <nanovdb/NanoVDB.h>

template <typename T>
__device__ inline T lerp(T v0, T v1, T t) {
	return fma(t, v1, fma(-t, v0, v0));
}

template <typename Func, typename... Args>
__global__ void lambdaKernel(const size_t numItems, Func func, Args... args) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numItems) return;
	func(tid, args...);
}

template <typename Func, typename... Args>
__global__ void lambdaKernelLeaf(const size_t leafCount, Func func, Args... args) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= leafCount * 512) return;
	func(tid, args...);
}

inline size_t blocksPerGrid(const size_t numItems, const size_t threadsPerBlock) {
	NANOVDB_ASSERT(numItems > 0 && threadsPerBlock >= 32 && threadsPerBlock % 32 == 0);
	return (numItems + threadsPerBlock - 1) / threadsPerBlock;
}