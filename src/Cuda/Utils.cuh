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

template <typename ValueT>
struct CudaResources {
	nanovdb::Coord* d_coords = nullptr;
	ValueT* d_values = nullptr;
	ValueT* d_temp_values = nullptr;
	cudaEvent_t beenCopied;
	bool initialized;

	__host__ CudaResources(const size_t npoints, const cudaStream_t& stream) : initialized(false) {
		cudaError_t err;

		err = cudaMallocAsync(&d_coords, npoints * sizeof(nanovdb::Coord), stream);
		if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_coords");

		err = cudaMallocAsync(&d_values, npoints * sizeof(ValueT), stream);
		if (err != cudaSuccess) {
			cudaFreeAsync(d_coords, stream);
			throw std::runtime_error("Failed to allocate d_values");
		}

		err = cudaMallocAsync(&d_temp_values, npoints * sizeof(ValueT), stream);
		if (err != cudaSuccess) {
			cudaFreeAsync(d_coords, stream);
			cudaFreeAsync(d_values, stream);
			throw std::runtime_error("Failed to allocate d_temp_values");
		}

		err = cudaEventCreateWithFlags(&beenCopied, cudaEventDisableTiming);
		if (err != cudaSuccess) {
			clear(stream);
			throw std::runtime_error("Failed to create CUDA event");
		}

		initialized = true;
		cudaEventRecord(beenCopied, stream);
	}

	// Remove the destructor and handle cleanup explicitly
	__host__ void cleanup(const cudaStream_t& stream) {
		if (initialized) {
			cudaStreamSynchronize(stream);
			clear(stream);
		}
	}

	__host__ void waitForInit(const cudaStream_t& stream) const {
		if (initialized) {
			cudaStreamWaitEvent(stream, beenCopied);
		}
	}

	__host__ __device__ bool isReady() const { return initialized && (cudaEventQuery(beenCopied) == cudaSuccess); }

	__host__ void clear(const cudaStream_t& stream) {
		if (initialized) {
			cudaStreamSynchronize(stream);
			cudaFreeAsync(d_coords, stream);
			cudaFreeAsync(d_values, stream);
			cudaFreeAsync(d_temp_values, stream);
			cudaEventDestroy(beenCopied);
			initialized = false;
		}
	}
};

template <typename ValueInT, typename ValueOutT>
struct HostMemoryManager {
	const HNS::OpenGrid<ValueInT>& in_data;
	HNS::NanoGrid<ValueOutT>& out_data;
	bool registered;

	HostMemoryManager(const HNS::OpenGrid<ValueInT>& in, HNS::NanoGrid<ValueOutT>& out)
	    : in_data(in), out_data(out), registered(false) {
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

template <typename ValueInT, typename ValueOutT>
void LoadPointData(CudaResources<ValueOutT>& resources, const HNS::OpenGrid<ValueInT>& in_data, const size_t npoints,
                   const cudaStream_t& stream) {
	resources.waitForInit(stream);

	cudaMemcpyAsync(resources.d_coords, (nanovdb::Coord*)in_data.pCoords(), npoints * sizeof(openvdb::Coord),
	                cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(resources.d_values, (ValueOutT*)in_data.pValues(), npoints * sizeof(ValueOutT),
	                cudaMemcpyHostToDevice, stream);

	cudaEventRecord(resources.beenCopied, stream);
	cudaStreamWaitEvent(stream, resources.beenCopied);
}

template <typename ValueT>
void UnloadPointData(CudaResources<ValueT>& resources, HNS::NanoGrid<ValueT>& out_data, const size_t npoints,
                     const cudaStream_t& stream) {
	cudaMemcpyAsync(out_data.pValues(), resources.d_temp_values, npoints * sizeof(ValueT), cudaMemcpyDeviceToHost,
	                stream);
	cudaMemcpyAsync(out_data.pCoords(), resources.d_coords, npoints * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost,
	                stream);

	cudaEventRecord(resources.beenCopied, stream);
	cudaStreamWaitEvent(stream, resources.beenCopied);
}