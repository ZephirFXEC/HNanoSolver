#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

void combustion(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const float dt,
                const float voxelSize, const cudaStream_t& mainStream) {
	const size_t totalVoxels = data.size();

	// Get velocity block (assuming exactly one Vec3f block)
	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	const auto floatBlocks = data.getBlocksOfType<float>();
	if (floatBlocks.empty()) {
		throw std::runtime_error("No float blocks found");
	}

	const float* density = data.pValues<float>(floatBlocks[0]);

	if (!velocity) {
		throw std::runtime_error("Velocity data not found");
	}
	if (!density) {
		throw std::runtime_error("Density data not found");
	}

	float* d_density = nullptr;
	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Vec3f* d_outVel = nullptr;
	nanovdb::Coord* d_coords = nullptr;

	cudaMalloc(&d_density, totalVoxels * sizeof(float));
	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_outVel, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));

	cudaMemcpy(d_density, density, totalVoxels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice);


	// Process each float block
	constexpr int blockSize = 256;
	const int numBlocks = (totalVoxels + blockSize - 1) / blockSize;
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	// vel_y_density<<<numBlocks, blockSize, 0, mainStream>>>(gpuGrid, d_velocity, d_density, d_outVel, totalVoxels);

	cudaMemcpyAsync(velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, mainStream);

	cudaStreamSynchronize(mainStream);

	cudaFree(d_density);
	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_outVel);
}

extern "C" void CombustionKernel(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const float dt,
                                 const float voxelSize, const cudaStream_t& stream) {
	combustion(data, handle, dt, voxelSize, stream);
}