//
// Created by zphrfx on 24/08/2024.
//

#ifndef CUDASTREAMWRAPPER_HPP
#define CUDASTREAMWRAPPER_HPP

#include <cuda_runtime.h>
#include <cstdio>

class CudaStreamWrapper {
public:
	CudaStreamWrapper() {
	 if (const cudaError_t err = cudaStreamCreate(&stream); err != cudaSuccess) {
			printf("Failed to create stream: %s\n", cudaGetErrorString(err));
			stream = nullptr;
		}
	}
	~CudaStreamWrapper() {
		if (stream) {
			cudaStreamDestroy(stream);
		}
	}

	[[nodiscard]] cudaStream_t getStream() const { return stream; }

private:
	cudaStream_t stream;
};


#endif //CUDASTREAMWRAPPER_HPP
