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

	cudaMallocAsync(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), stream);
	cudaMallocAsync(&d_coords, totalVoxels * sizeof(nanovdb::Coord), stream);
	cudaMallocAsync(&d_divergence, totalVoxels * sizeof(float), stream);
	cudaMallocAsync(&d_pressure, totalVoxels * sizeof(float), stream);

	// Use async memory operations with stream
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);
	cudaMemsetAsync(d_divergence, 0, totalVoxels * sizeof(float), stream);
	cudaMemsetAsync(d_pressure, 0, totalVoxels * sizeof(float), stream);

	// Create grid handle and get device grid pointer
	auto handle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);

	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();


	int numLeaves = totalVoxels / 512;
	dim3 numBrick(8, 8, 8);
	{
		ScopedTimerGPU timer("HNanoSolver::Divergence", 12 /* Vec3f */ + 4 /* float */, totalVoxels);

		divergence_opt<<<numLeaves, numBrick, 0, stream>>>(gpuGrid, d_velocity, d_divergence, 1.0f / voxelSize, numLeaves);
	}
	// Red-black Gauss-Seidel iterations
	{
		ScopedTimerGPU timer("HNanoSolver::Pressure", 4 * 2 /* float */, totalVoxels * iteration);
		const float omega = 2.0f / (1.0f + sin(3.14159 * voxelSize));
		for (int iter = 0; iter < iteration; ++iter) {
			redBlackGaussSeidelUpdate_opt<<<numLeaves, numBrick, 0, stream>>>(gpuGrid, d_divergence, d_pressure, voxelSize, totalVoxels, 0,
			                                                                  omega);
			redBlackGaussSeidelUpdate_opt<<<numLeaves, numBrick, 0, stream>>>(gpuGrid, d_divergence, d_pressure, voxelSize, totalVoxels, 1,
			                                                                  omega);
		}
	}

	{
		ScopedTimerGPU timer("HNanoSolver::Projection", 12 * 2 /* Vec3f */ + 4 /* float */, totalVoxels);
		subtractPressureGradient_opt<<<numLeaves, numBrick, 0, stream>>>(gpuGrid, d_velocity, d_pressure, d_velocity, 1.0f / voxelSize,
		                                                                 numLeaves);
	}

	// Copy result back asynchronously
	cudaMemcpyAsync(velocity, d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, stream);

	// Clean up resources
	cudaStreamSynchronize(stream);

	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_divergence);
	cudaFree(d_pressure);
}


void divergence(HNS::GridIndexedData& data, const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();
	const size_t numLeaves = totalVoxels / 512;

	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));
	float* h_divergence = data.pValues<float>("divergence");

	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	float* d_divergence = nullptr;

	cudaMallocAsync(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), stream);
	cudaMallocAsync(&d_coords, totalVoxels * sizeof(nanovdb::Coord), stream);
	cudaMallocAsync(&d_divergence, totalVoxels * sizeof(float), stream);

	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);

	cudaMemsetAsync(d_divergence, 0, totalVoxels * sizeof(float), stream);

	cudaStreamSynchronize(stream);

	auto handle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();
	int blockSize = 256;
	const int gridSize = (totalVoxels + blockSize - 1) / blockSize;
	{
		ScopedTimerGPU("HNanoSolver::Divergence", 12 + 4, totalVoxels);
		divergence<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_divergence, 1.0f / voxelSize, totalVoxels);
	}

	cudaMemcpyAsync(h_divergence, d_divergence, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, stream);

	// Clean up resources
	cudaStreamSynchronize(stream);

	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_divergence);
}

extern "C" void Divergence(HNS::GridIndexedData& data, const float voxelSize, const cudaStream_t& stream) {
	divergence(data, voxelSize, stream);
}


extern "C" void ProjectNonDivergent(HNS::GridIndexedData& data, const size_t iterations, const float voxelSize,
                                    const cudaStream_t& stream) {
	pressure_projection_idx(data, iterations, voxelSize, stream);
}