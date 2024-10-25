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
struct PinnedAllocator {
	static void allocate(T* ptr, const size_t n) { cudaMallocHost(&ptr, n * sizeof(T)); };
	static void deallocate(T* ptr) { cudaFreeHost(ptr); }
};

template <typename ValueT>
struct CudaResources {
	nanovdb::Coord* d_coords = nullptr;
	ValueT* d_values = nullptr;
	ValueT* d_temp_values = nullptr;
	cudaEvent_t beenCopied;

	CudaResources(const size_t npoints, const cudaStream_t& stream) {
		cudaError_t err;

		err = cudaMallocAsync(&d_coords, npoints * sizeof(nanovdb::Coord), stream);
		if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_coords");

		err = cudaMallocAsync(&d_values, npoints * sizeof(ValueT), stream);
		if (err != cudaSuccess) {
			cudaFreeAsync(d_coords, stream);  // Clean up previously allocated resources
			throw std::runtime_error("Failed to allocate d_values");
		}

		err = cudaMallocAsync(&d_temp_values, npoints * sizeof(ValueT), stream);
		if (err != cudaSuccess) {
			cudaFreeAsync(d_coords, stream);
			cudaFreeAsync(d_values, stream);
			throw std::runtime_error("Failed to allocate d_temp_values");
		}

		// Create event with error checking
		err = cudaEventCreateWithFlags(&beenCopied, cudaEventDisableTiming);
		if (err != cudaSuccess) {
			clear(stream);  // Clean up all resources if event creation fails
			throw std::runtime_error("Failed to create CUDA event");
		}

		// Record event indicating that resources are initialized
		cudaEventRecord(beenCopied, stream);
	}

	void waitForInit(const cudaStream_t& stream) const { cudaStreamWaitEvent(stream, beenCopied); }

	bool isReady() const { return cudaEventQuery(beenCopied) == cudaSuccess; }

	void clear(const cudaStream_t& stream) const {
		// Free device memory asynchronously
		if (d_coords) cudaFreeAsync(d_coords, stream);
		if (d_values) cudaFreeAsync(d_values, stream);
		if (d_temp_values) cudaFreeAsync(d_temp_values, stream);

		// Destroy event
		cudaEventDestroy(beenCopied);
	}
};

template <typename ValueInT, typename ValueOutT>
struct HostMemoryManager {
	const HNS::OpenGrid<ValueInT>& in_data;
	HNS::NanoGrid<ValueOutT>& out_data;

	HostMemoryManager(const HNS::OpenGrid<ValueInT>& in, HNS::NanoGrid<ValueOutT>& out) : in_data(in), out_data(out) {
		if (in_data.size != 0) {
			cudaHostRegister(in_data.pCoords, in_data.size * sizeof(openvdb::Coord), cudaHostRegisterDefault);
			cudaHostRegister(in_data.pValues, in_data.size * sizeof(ValueInT), cudaHostRegisterDefault);
		}

		if (out_data.size != 0) {
			cudaHostRegister(out_data.pCoords, out_data.size * sizeof(nanovdb::Coord), cudaHostRegisterDefault);
			cudaHostRegister(out_data.pValues, out_data.size * sizeof(ValueOutT), cudaHostRegisterDefault);
		}
	}

	~HostMemoryManager() {
			cudaHostUnregister(in_data.pCoords);
			cudaHostUnregister(in_data.pValues);
			cudaHostUnregister(out_data.pCoords);
			cudaHostUnregister(out_data.pValues);
	}
};

template <typename ValueInT, typename ValueOutT>
void LoadPointData(CudaResources<ValueOutT>& resources, const HNS::OpenGrid<ValueInT>& in_data, const size_t npoints,
                   const cudaStream_t& stream) {
	resources.waitForInit(stream);

	cudaMemcpyAsync(resources.d_coords, (nanovdb::Coord*)in_data.pCoords, npoints * sizeof(openvdb::Coord),
	                cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(resources.d_values, (ValueOutT*)in_data.pValues, npoints * sizeof(ValueOutT),
	                cudaMemcpyHostToDevice, stream);

	cudaEventRecord(resources.beenCopied, stream);
	cudaStreamWaitEvent(stream, resources.beenCopied);
}

template <typename ValueT>
void UnloadPointData(CudaResources<ValueT>& resources, HNS::NanoGrid<ValueT>& out_data, const size_t npoints,
                     const cudaStream_t& stream) {
	out_data.size = npoints;

	if (!out_data.pCoords) {
		out_data.pCoords = new nanovdb::Coord[npoints];
	}
	if (!out_data.pValues) {
		out_data.pValues = new ValueT[npoints];
	}

	cudaMemcpyAsync(out_data.pValues, resources.d_temp_values, npoints * sizeof(ValueT), cudaMemcpyDeviceToHost,
	                stream);
	cudaMemcpyAsync(out_data.pCoords, resources.d_coords, npoints * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost,
	                stream);

	cudaEventRecord(resources.beenCopied, stream);
	cudaStreamWaitEvent(stream, resources.beenCopied);
}