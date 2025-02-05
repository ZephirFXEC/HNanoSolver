#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>

#include "../Utils/GridData.hpp"

template <typename T>
__device__ T lerp(T v0, T v1, T t) {
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


template <typename CoordT, typename ValueT, bool HasTemp>
struct CudaResources;


// Specialization when HasTemp is true
template <typename CoordT, typename ValueT>
struct CudaResources<CoordT, ValueT, true> {
	CoordT* d_coords = nullptr;
	ValueT* d_values = nullptr;
	ValueT* d_temp_values = nullptr;

	__host__ CudaResources(const size_t npoints, const cudaStream_t& stream) {
		// Allocate device memory asynchronously
		cudaCheck(cudaMallocAsync(&d_coords, npoints * sizeof(CoordT), stream));
		cudaCheck(cudaMallocAsync(&d_values, npoints * sizeof(ValueT), stream));
		cudaCheck(cudaMallocAsync(&d_temp_values, npoints * sizeof(ValueT), stream));
		// No need to synchronize here; subsequent operations will ensure proper sequencing
	}

	template <typename ValueInT>
	__host__ void LoadPointData(const HNS::OpenGrid<ValueInT>& in_data, const cudaStream_t& stream) {
		// Copy data from host to device asynchronously
		cudaCheck(cudaMemcpyAsync(d_coords, in_data.pCoords(), in_data.size * sizeof(CoordT), cudaMemcpyHostToDevice, stream));
		cudaCheck(cudaMemcpyAsync(d_values, in_data.pValues(), in_data.size * sizeof(ValueInT), cudaMemcpyHostToDevice, stream));
	}

	__host__ void LoadPointCoord(const openvdb::Coord* in_data, const size_t size, const cudaStream_t& stream) {
		cudaCheck(cudaMemcpyAsync(d_coords, in_data, size * sizeof(CoordT), cudaMemcpyHostToDevice, stream));
	}

	template <typename ValueInT>
	__host__ void LoadPointValue(const ValueInT* in_data, const size_t size, const cudaStream_t& stream) {
		cudaCheck(cudaMemcpyAsync(d_values, in_data, size * sizeof(ValueT), cudaMemcpyHostToDevice, stream));
	}

	template <typename ValueOutT>
	__host__ void UnloadPointData(HNS::NanoGrid<ValueOutT>& out_data, const cudaStream_t& stream) {
		// Copy data from device to host asynchronously
		cudaCheck(cudaMemcpyAsync(out_data.pCoords(), d_coords, out_data.size * sizeof(CoordT), cudaMemcpyDeviceToHost, stream));
		cudaCheck(cudaMemcpyAsync(out_data.pValues(), d_temp_values, out_data.size * sizeof(ValueOutT), cudaMemcpyDeviceToHost, stream));
	}

	__host__ void cleanup(const cudaStream_t& stream) {
		// Synchronize the stream to ensure all operations are complete
		cudaCheck(cudaStreamSynchronize(stream));
		clear();
	}

	__host__ void clear() {
		if (d_coords) cudaCheck(cudaFree(d_coords));
		if (d_values) cudaCheck(cudaFree(d_values));
		if (d_temp_values) cudaCheck(cudaFree(d_temp_values));
		d_coords = nullptr;
		d_values = nullptr;
		d_temp_values = nullptr;
	}
};

// Specialization when HasTemp is false
template <typename CoordT, typename ValueT>
struct CudaResources<CoordT, ValueT, false> {
	CoordT* d_coords = nullptr;
	ValueT* d_values = nullptr;

	__host__ CudaResources(const size_t npoints, const cudaStream_t& stream) {
		// Allocate device memory asynchronously
		cudaCheck(cudaMallocAsync(&d_coords, npoints * sizeof(CoordT), stream));
		cudaCheck(cudaMallocAsync(&d_values, npoints * sizeof(ValueT), stream));
	}

	template <typename ValueInT>
	__host__ void LoadPointData(const HNS::OpenGrid<ValueInT>& in_data, const cudaStream_t& stream) {
		// Copy data from host to device asynchronously
		cudaCheck(cudaMemcpyAsync(d_coords, in_data.pCoords(), in_data.size * sizeof(CoordT), cudaMemcpyHostToDevice, stream));
		cudaCheck(cudaMemcpyAsync(d_values, in_data.pValues(), in_data.size * sizeof(ValueInT), cudaMemcpyHostToDevice, stream));
	}

	__host__ void LoadPointData(const HNS::GridIndexedData& in_data, const cudaStream_t& stream) const {
		// Copy data from host to device asynchronously
		cudaCheck(cudaMemcpyAsync(d_coords, in_data.pCoords(), in_data.size() * sizeof(CoordT), cudaMemcpyHostToDevice, stream));
		cudaCheck(cudaMemcpyAsync(d_values, in_data.pValues<ValueT>("density"), in_data.size() * sizeof(ValueT), cudaMemcpyHostToDevice, stream));
	}

	__host__ void LoadPointCoord(const openvdb::Coord* in_data, const size_t size, const cudaStream_t& stream) const {
		cudaCheck(cudaMemcpyAsync(d_coords, in_data, size * sizeof(CoordT), cudaMemcpyHostToDevice, stream));
	}

	template <typename ValueInT>
	__host__ void LoadPointValue(const ValueInT* in_data, const size_t size, const cudaStream_t& stream) {
		cudaCheck(cudaMemcpyAsync(d_values, in_data, size * sizeof(ValueT), cudaMemcpyHostToDevice, stream));
	}

	template <typename ValueOutT>
	__host__ void UnloadPointData(HNS::NanoGrid<ValueOutT>& out_data, const cudaStream_t& stream) {
		// Copy data from device to host asynchronously
		cudaCheck(cudaMemcpyAsync(out_data.pCoords(), d_coords, out_data.size * sizeof(CoordT), cudaMemcpyDeviceToHost, stream));
		cudaCheck(cudaMemcpyAsync(out_data.pValues(), d_values, out_data.size * sizeof(ValueOutT), cudaMemcpyDeviceToHost, stream));
	}

	__host__ void cleanup(const cudaStream_t& stream) {
		// Synchronize the stream to ensure all operations are complete
		cudaCheck(cudaStreamSynchronize(stream));
		clear();
	}

	__host__ void clear() {
		if (d_coords) cudaCheck(cudaFree(d_coords));
		if (d_values) cudaCheck(cudaFree(d_values));
		d_coords = nullptr;
		d_values = nullptr;
	}
};


template <typename T, typename U = std::conditional_t<std::is_same_v<T, float>, nanovdb::FloatTree, nanovdb::Vec3fTree>,
          bool HasTemp = true>
__global__ void set_grid_values(const CudaResources<nanovdb::Coord, T, HasTemp> ressources, const size_t npoints, nanovdb::Grid<U>* __restrict__ d_grid) {
	auto accessor = d_grid->tree().getAccessor();

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	nanovdb::Coord coord = ressources.d_coords[tid];
	T value = ressources.d_values[tid];

	accessor.template set<nanovdb::SetVoxel<T>>(coord, value);
}

template <typename T, typename U = std::conditional_t<std::is_same_v<T, float>, nanovdb::FloatTree, nanovdb::Vec3fTree>,
          bool HasTemp = true>
__global__ void zero_init_grid(const CudaResources<nanovdb::Coord, T, HasTemp> resources, const size_t npoints, nanovdb::Grid<U>* __restrict__ d_grid) {
	auto accessor = d_grid->tree().getAccessor();

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	nanovdb::Coord coord = resources.d_coords[tid];
	T value = T(0);

	accessor.template set<nanovdb::SetVoxel<T>>(coord, value);
}

template <typename T, typename U = std::conditional_t<std::is_same_v<T, float>, nanovdb::FloatTree, nanovdb::Vec3fTree>>
__global__ void get_grid_values(const CudaResources<nanovdb::Coord, T, true> ressources, const size_t npoints, nanovdb::Grid<U>* __restrict__ d_grid) {
	auto accessor = d_grid->tree().getAccessor();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const nanovdb::Coord coord = ressources.d_coords[tid];

	ressources.d_temp_values[tid] = accessor.getValue(coord);
}

inline __device__ nanovdb::Vec3f sampleMACVelocity(
    const decltype(nanovdb::math::createSampler<1>(
        std::declval<const decltype(std::declval<nanovdb::Vec3fGrid>().tree().getAccessor())>()))& velSampler,
    const nanovdb::Vec3f& pos) {
	const float u = velSampler(pos + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float v = velSampler(pos + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float w = velSampler(pos + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	return {u, v, w};
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