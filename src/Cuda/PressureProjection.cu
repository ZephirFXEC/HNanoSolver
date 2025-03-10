#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"


void pressure_projection_idx(HNS::GridIndexedData& data, const size_t iteration, const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();
	constexpr int blockSize = 256;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;

	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	float* d_divergence = nullptr;
	float* d_pressure = nullptr;

	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMalloc(&d_divergence, totalVoxels * sizeof(float));
	cudaMalloc(&d_pressure, totalVoxels * sizeof(float));

	// Use async memory operations with stream
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);
	cudaMemsetAsync(d_divergence, 0, totalVoxels * sizeof(float), stream);
	cudaMemsetAsync(d_pressure, 0, totalVoxels * sizeof(float), stream);

	// Create grid handle and get device grid pointer
	auto handle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);

	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	divergence<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_divergence, voxelSize, totalVoxels);

	// Red-black Gauss-Seidel iterations
	constexpr float omega = 1.9f;  // SOR relaxation parameter
	for (int iter = 0; iter < iteration; ++iter) {
		redBlackGaussSeidelUpdate<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize, totalVoxels,
		                                                               0, omega);
		redBlackGaussSeidelUpdate<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize, totalVoxels,
		                                                               1, omega);
	}

	// Apply pressure gradient
	subtractPressureGradient<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, totalVoxels, d_velocity, d_pressure, d_velocity,
	                                                              voxelSize);

	// Copy result back asynchronously
	cudaMemcpyAsync(velocity, d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, stream);

	// Clean up resources
	cudaStreamSynchronize(stream);

	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_divergence);
	cudaFree(d_pressure);
}


void pressure_projection(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const size_t iteration,
                         const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();
	constexpr int blockSize = 256;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;

	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	float* d_divergence = nullptr;
	float* d_pressure = nullptr;

	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMalloc(&d_divergence, totalVoxels * sizeof(float));
	cudaMalloc(&d_pressure, totalVoxels * sizeof(float));

	// Use async memory operations with stream
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);
	cudaMemsetAsync(d_divergence, 0, totalVoxels * sizeof(float), stream);
	cudaMemsetAsync(d_pressure, 0, totalVoxels * sizeof(float), stream);

	// Create grid handle and get device grid pointer
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	divergence<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_divergence, voxelSize, totalVoxels);

	// Red-black Gauss-Seidel iterations
	constexpr float omega = 1.9f;  // SOR relaxation parameter
	for (int iter = 0; iter < iteration; ++iter) {
		redBlackGaussSeidelUpdate<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize, totalVoxels,
		                                                               0, omega);
		redBlackGaussSeidelUpdate<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize, totalVoxels,
		                                                               1, omega);
	}

	// Apply pressure gradient
	subtractPressureGradient<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, totalVoxels, d_velocity, d_pressure, d_velocity,
	                                                              voxelSize);

	// Copy result back asynchronously
	cudaMemcpyAsync(velocity, d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, stream);

	// Clean up resources
	cudaStreamSynchronize(stream);

	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_divergence);
	cudaFree(d_pressure);
}

extern "C" void Divergence_idx(HNS::GridIndexedData& data, const size_t iterations, const float voxelSize, const cudaStream_t& stream) {
	pressure_projection_idx(data, iterations, voxelSize, stream);
}

extern "C" void Project(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const size_t iterations,
                        const float voxelSize, const cudaStream_t& stream) {
	pressure_projection(data, handle, iterations, voxelSize, stream);
}