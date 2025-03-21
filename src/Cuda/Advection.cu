#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

// Enable L1 caching of global memory with compiler hints
#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 186

void advect_index_grid(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& mainStream) {
	const size_t totalVoxels = data.size();
	const float inv_voxelSize = 1.0f / voxelSize;

	// Get velocity block (assuming exactly one Vec3f block)
	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	const nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	// Get all float blocks (density, temperature, fuel, etc.)
	const auto floatBlocks = data.getBlocksOfType<float>();
	if (floatBlocks.empty()) {
		throw std::runtime_error("No float blocks found");
	}

	// Validate pointers
	if (!velocity) {
		throw std::runtime_error("Velocity data not found");
	}


	// Create streams and storage for float blocks
	std::vector<cudaStream_t> streams(floatBlocks.size());
	std::vector<float*> hostPointers;
	std::vector<float*> d_inputs, d_outputs;

	// Initialize resources for each float block
	for (auto& name : floatBlocks) {
		cudaStreamCreate(&streams[hostPointers.size()]);

		auto* host_ptr = data.pValues<float>(name);
		if (!host_ptr) {
			throw std::runtime_error("Block '" + name + "' not found or type mismatch");
		}
		hostPointers.push_back(host_ptr);

		float *d_in, *d_out;
		cudaMalloc(&d_in, totalVoxels * sizeof(float));
		cudaMalloc(&d_out, totalVoxels * sizeof(float));
		d_inputs.push_back(d_in);
		d_outputs.push_back(d_out);
	}


	// Allocate shared device memory
	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));

	// Copy shared data
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, mainStream);
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, mainStream);


	auto gridHandle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);
	const auto gpuGrid = gridHandle.deviceGrid<nanovdb::ValueOnIndex>();

	int deviceId, blockSize, minGridSize;
	cudaGetDevice(&deviceId);
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, advect_scalar, 0, 0);

	// Calculate grid dimensions based on optimal block size
	const int gridSize = (totalVoxels + blockSize - 1) / blockSize;

	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		cudaMemcpyAsync(d_inputs[i], hostPointers[i], totalVoxels * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
	}

	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Scalar", 8 + 12 + 4 + 4, totalVoxels);

		for (size_t i = 0; i < floatBlocks.size(); ++i) {
			advect_scalar<<<gridSize, blockSize, 0, streams[i]>>>(gpuGrid, d_coords, d_velocity, d_inputs[i], d_outputs[i], totalVoxels, dt,
			                                                      inv_voxelSize);
		}
	}

	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		cudaMemcpyAsync(hostPointers[i], d_outputs[i], totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}

	// Synchronize and cleanup
	cudaStreamSynchronize(mainStream);
	for (auto& stream : streams) {
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}

	// Free device memory
	cudaFree(d_velocity);
	cudaFree(d_coords);
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		cudaFree(d_inputs[i]);
		cudaFree(d_outputs[i]);
	}
}

void advect_index_grid_v(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();
	const float inv_voxelSize = 1.0f / voxelSize;

	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	// Allocate and initialize device memory
	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	nanovdb::Vec3f* d_outVel = nullptr;

	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMalloc(&d_outVel, totalVoxels * sizeof(nanovdb::Vec3f));

	// Prefill with zeros to avoid page faults during kernel execution
	cudaMemsetAsync(d_outVel, 0, totalVoxels * sizeof(nanovdb::Vec3f), stream);

	// Copy data asynchronously
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);

	// Create grid and prepare for kernel launch
	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle =
	    nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	int deviceId, blockSize, minGridSize;
	cudaGetDevice(&deviceId);
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, advect_vector, 0, 0);
	const int gridSize = (totalVoxels + blockSize - 1) / blockSize;

	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Velocity", 8 + 12 + 12, totalVoxels);
		advect_vector<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_outVel, totalVoxels, dt, inv_voxelSize);
	}

	// Make sure kernel is finished before copying back
	cudaStreamSynchronize(stream);

	// Use synchronous copy for non-pinned memory
	cudaMemcpy(velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost);

	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_outVel);
}


extern "C" void AdvectIndexGrid(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& stream) {
	advect_index_grid(data, dt, voxelSize, stream);
}

extern "C" void AdvectIndexGridVelocity(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& stream) {
	advect_index_grid_v(data, dt, voxelSize, stream);
}
