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

template <typename T>
inline __device__ T computeError(const T& value, const T& value_backward) {
	return static_cast<T>(0.5f) * (value - value_backward);
}

inline __device__ float absValue(const float x) {
	return fabsf(x);
}

inline __device__ nanovdb::Vec3f absValue(const nanovdb::Vec3f& v) {
	return nanovdb::Vec3f(fabsf(v[0]), fabsf(v[1]), fabsf(v[2]));
}

template <typename T>
inline __device__ T computeMaxCorrection(const T& value_forward, const T& value) {
	return static_cast<T>(0.5f) * absValue(value_forward - value);
}

inline __device__ float clampValue(const float x, const float minVal, const float maxVal) {
	return fminf(fmaxf(x, minVal), maxVal);
}

inline __device__ nanovdb::Vec3f clampValue(const nanovdb::Vec3f& x, const nanovdb::Vec3f& minVal, const nanovdb::Vec3f& maxVal) {
	return nanovdb::Vec3f(
	    fminf(fmaxf(x[0], minVal[0]), maxVal[0]),
	    fminf(fmaxf(x[1], minVal[1]), maxVal[1]),
	    fminf(fmaxf(x[2], minVal[2]), maxVal[2])
	);
}

inline __device__ float lerp(const float a, const float b, const float t) {
	return a + t * (b - a);
}

inline __device__ nanovdb::Vec3f lerp(const nanovdb::Vec3f& a, const nanovdb::Vec3f& b, const float t) {
	return a + (b - a) * t;
}

inline __device__ float enforceNonNegative(const float x) {
	return fmaxf(0.0f, x);
}

inline __device__ nanovdb::Vec3f enforceNonNegative(const nanovdb::Vec3f& v) {
	return v;
}

template <typename ValueT>
struct CudaResources {
	nanovdb::Coord* d_coords = nullptr;
	ValueT* d_values = nullptr;
	ValueT* d_temp_values = nullptr;
	cudaEvent_t ValueBeenCopied, ValueBeenMalloced, CoordBeenCopied, CoordBeenMalloced;

	__host__ CudaResources(const size_t npoints, const cudaStream_t& stream) {

		cudaCheck(cudaEventCreateWithFlags(&CoordBeenMalloced, cudaEventDisableTiming));
		cudaCheck(cudaEventCreateWithFlags(&ValueBeenMalloced, cudaEventDisableTiming));
		cudaCheck(cudaEventCreateWithFlags(&CoordBeenCopied, cudaEventDisableTiming));
		cudaCheck(cudaEventCreateWithFlags(&ValueBeenCopied, cudaEventDisableTiming));

		cudaCheck(cudaMallocAsync(&d_coords, npoints * sizeof(nanovdb::Coord), stream));
		cudaCheck(cudaEventRecord(CoordBeenMalloced, stream));

		cudaCheck(cudaMallocAsync(&d_values, npoints * sizeof(ValueT), stream));
		cudaCheck(cudaEventRecord(ValueBeenMalloced, stream));

		cudaCheck(cudaMallocAsync(&d_temp_values, npoints * sizeof(ValueT), stream));
	}

	template <typename ValueInT>
	__host__ void LoadPointData(const HNS::OpenGrid<ValueInT>& in_data, const cudaStream_t& stream) {
		cudaCheck(cudaStreamWaitEvent(stream, CoordBeenMalloced, 0));
		cudaCheck(cudaStreamWaitEvent(stream, ValueBeenMalloced, 0));

		cudaCheck(cudaMemcpyAsync(d_coords, (nanovdb::Coord*)in_data.pCoords(), in_data.size * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream));
		cudaCheck(cudaEventRecord(CoordBeenCopied, stream));

		cudaCheck(cudaMemcpyAsync(d_values, (ValueT*)in_data.pValues(), in_data.size * sizeof(ValueT), cudaMemcpyHostToDevice, stream));
		cudaCheck(cudaEventRecord(ValueBeenCopied, stream));
	}

	__host__ void UnloadPointData(HNS::NanoGrid<ValueT>& out_data, const cudaStream_t& stream) {
		cudaCheck(cudaMemcpyAsync(out_data.pValues(), d_temp_values, out_data.size * sizeof(ValueT), cudaMemcpyDeviceToHost, stream));
		cudaCheck(cudaMemcpyAsync(out_data.pCoords(), d_coords, out_data.size * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost, stream));
	}

	__host__ void cleanup(const cudaStream_t& stream) const {
		cudaCheck(cudaStreamSynchronize(stream));
		clear(stream);
	}

	__host__ void clear(const cudaStream_t& stream) const {
		if (d_coords) cudaFreeAsync(d_coords, stream);
		if (d_values) cudaFreeAsync(d_values, stream);
		if (d_temp_values) cudaFreeAsync(d_temp_values, stream);

		cudaCheck(cudaEventDestroy(CoordBeenMalloced));
		cudaCheck(cudaEventDestroy(ValueBeenMalloced));
		cudaCheck(cudaEventDestroy(CoordBeenCopied));
		cudaCheck(cudaEventDestroy(ValueBeenCopied));
	}
};

template <typename ValueInT, typename ValueOutT>
struct HostMemoryManager {
	const HNS::OpenGrid<ValueInT>& in_data;
	HNS::NanoGrid<ValueOutT>& out_data;
	bool registered;

	HostMemoryManager(const HNS::OpenGrid<ValueInT>& in, HNS::NanoGrid<ValueOutT>& out) : in_data(in), out_data(out), registered(false) {
		cudaError_t err;

		// Register input memory
		err = cudaHostRegister((void*)in_data.pCoords(), in_data.size * sizeof(openvdb::Coord), cudaHostRegisterReadOnly);
		if (err != cudaSuccess) throw std::runtime_error("Failed to register input coords");

		err = cudaHostRegister((void*)in_data.pValues(), in_data.size * sizeof(ValueInT), cudaHostRegisterReadOnly);
		if (err != cudaSuccess) {
			cudaHostUnregister((void*)in_data.pCoords());
			throw std::runtime_error("Failed to register input values");
		}

		registered = true;
	}

	~HostMemoryManager() {
		if (registered) {
			cudaHostUnregister((void*)in_data.pCoords());
			cudaHostUnregister((void*)in_data.pValues());
		}
	}

	// Delete copy constructor and assignment operator
	HostMemoryManager(const HostMemoryManager&) = delete;
	HostMemoryManager& operator=(const HostMemoryManager&) = delete;
};
