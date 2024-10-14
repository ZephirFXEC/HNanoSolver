#pragma once

#include <nanovdb/NanoVDB.h>

__device__ inline int lower_offset(const int i, const int j, const int k) {
	auto a = [](const int n) { return (n & 127) >> 3; };
	return a(i) << 8 | a(j) << 4 | a(k);  // 0 ,1 ,.. ,16^3 -1
};
__device__ inline int upper_offset(const int i, const int j, const int k) {
	auto a = [](const int n) { return (n & 4095) >> 7; };
	return a(i) << 10 | a(j) << 5 | a(k);  // 0 ,1 ,.. ,32^3 -1
};

__device__ inline int leaf_offset(const int i, const int j, const int k) {
	return (i & 7) << 6 | (j & 7) << 3 | k & 7;
}

__device__ inline uint64_t generateKeyA(const int i, const int j, const int k) {
	return (static_cast<uint64_t>(k) >> 12) | (static_cast<uint64_t>(j) >> 12) << 21 |
	       (static_cast<uint64_t>(i) >> 12) << 42;
}

__device__ inline uint64_t generateKeyB(const int i, const int j, const int k, const int M) {
	// Simplified offsets for demonstration
	const uint32_t upper_off = upper_offset(i, j, k);
	const uint32_t lower_off = lower_offset(i, j, k);
	const uint32_t leaf_off = leaf_offset(i, j, k);

	return static_cast<uint64_t>(M) << 36 | static_cast<uint64_t>(upper_off) << 21 |
	       static_cast<uint64_t>(lower_off) << 9 | static_cast<uint64_t>(leaf_off);
}

__global__ void inline generateKeys(const int* coords, uint64_t* keysA, uint64_t* keysB, const int N) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		const int i = coords[idx * 3];
		const int j = coords[idx * 3 + 1];
		const int k = coords[idx * 3 + 2];
		keysA[idx] = generateKeyA(i, j, k);
		keysB[idx] = generateKeyB(i, j, k, 0);  // M is initially 0
	}
}


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

inline size_t blocksPerGrid(const size_t numItems, const size_t threadsPerBlock) {
	NANOVDB_ASSERT(numItems > 0 && threadsPerBlock >= 32 && threadsPerBlock % 32 == 0);
	return (numItems + threadsPerBlock - 1) / threadsPerBlock;
}