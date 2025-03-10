#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/NanoVDB.h"


void Compute(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const int iteration,
             const float dt, const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();
	const float inv_voxelSize = 1.0f / voxelSize;

	// Get velocity block (assuming exactly one Vec3f block)
	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));
	if (!velocity) {
		throw std::runtime_error("Velocity data not found");
	}

	// Get all float blocks (density, temperature, fuel, etc.)
	const auto floatBlocks = data.getBlocksOfType<float>();
	if (floatBlocks.empty()) {
		throw std::runtime_error("No float blocks found");
	}

	// Create CUDA resources
	std::vector<cudaStream_t> streams(floatBlocks.size());
	std::vector<float*> hostPointers(floatBlocks.size());
	std::vector<float*> d_inputs(floatBlocks.size());
	std::vector<float*> d_outputs(floatBlocks.size());

	// Initialize resources for each float block
	for (size_t i = 0; i < floatBlocks.size(); i++) {
		cudaStreamCreate(&streams[i]);

		auto* host_ptr = data.pValues<float>(floatBlocks[i]);
		if (!host_ptr) {
			throw std::runtime_error("Block '" + floatBlocks[i] + "' not found or type mismatch");
		}
		hostPointers[i] = host_ptr;

		cudaMalloc(&d_inputs[i], totalVoxels * sizeof(float));
		cudaMalloc(&d_outputs[i], totalVoxels * sizeof(float));

		// Copy all scalar fields to device (fixing the bug where only the first field was copied)
		cudaMemcpyAsync(d_inputs[i], hostPointers[i], totalVoxels * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
	}

	// Allocate and initialize device memory
	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	nanovdb::Vec3f* d_outVel = nullptr;
	float* d_divergence = nullptr;
	float* d_pressure = nullptr;

	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMalloc(&d_outVel, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_divergence, totalVoxels * sizeof(float));
	cudaMalloc(&d_pressure, totalVoxels * sizeof(float));

	cudaMemsetAsync(d_outVel, 0, totalVoxels * sizeof(nanovdb::Vec3f), stream);
	cudaMemsetAsync(d_divergence, 0, totalVoxels * sizeof(float), stream);
	cudaMemsetAsync(d_pressure, 0, totalVoxels * sizeof(float), stream);

	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);

	// Create grid and prepare for kernel launch
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	// Calculate optimal kernel launch parameters
	int deviceId, blockSize, minGridSize;
	cudaGetDevice(&deviceId);
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, advect_vector, 0, 0);
	const int gridSize = (totalVoxels + blockSize - 1) / blockSize;


	// Simulation pipeline with synchronization points

	// Step 1: Advect velocity field
	advect_vector<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_outVel, totalVoxels, dt, voxelSize);
	cudaMemcpyAsync(d_velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToDevice, stream);

	// Step 2: Apply buoyancy forces
	vel_y_density<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_inputs[0], d_outVel, totalVoxels);
	cudaMemcpyAsync(d_velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToDevice, stream);

	// Step 3: Calculate velocity field divergence
	divergence<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_divergence, voxelSize, totalVoxels);

	// Step 4: Pressure solver (Red-black Gauss-Seidel iterations)
	constexpr float omega = 1.9f;  // SOR relaxation parameter
	for (int iter = 0; iter < iteration; ++iter) {
		redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize, totalVoxels,
		                                                              0, omega);

		redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize, totalVoxels,
		                                                              1, omega);
	}

	// Step 5: Apply pressure gradient to enforce incompressibility
	subtractPressureGradient<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, totalVoxels, d_velocity, d_pressure, d_velocity,
	                                                             voxelSize);

	// Sync before advecting scalar fields
	cudaStreamSynchronize(stream);

	// Step 6: Advect all scalar fields in parallel using individual streams
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		advect_scalar<<<gridSize, blockSize, 0, streams[i]>>>(gpuGrid, d_coords, d_velocity, d_inputs[i], d_outputs[i], totalVoxels, dt,
		                                                      inv_voxelSize);

		// Copy results back to host
		cudaMemcpyAsync(hostPointers[i], d_outputs[i], totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}

	// Synchronize all streams
	for (auto& s : streams) {
		cudaStreamSynchronize(s);
	}

	cudaMemcpy(velocity, d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost);

	// Clean up all allocated resources
	// Free device memory
	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_outVel);
	cudaFree(d_divergence);
	cudaFree(d_pressure);

	// Free scalar field resources
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		cudaFree(d_inputs[i]);
		cudaFree(d_outputs[i]);
		cudaStreamDestroy(streams[i]);
	}
}

void create_index_grid(HNS::GridIndexedData& data, nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const float voxelSize) {
	const auto* h_coords = data.pCoords();
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_coords, data.size() * sizeof(nanovdb::Coord));
	cudaMemcpy(d_coords, h_coords, data.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice);

	handle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);

	cudaFree(d_coords);
}

extern "C" void CreateIndexGrid(HNS::GridIndexedData& data, nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle,
                                const float voxelSize) {
	create_index_grid(data, handle, voxelSize);
}


extern "C" void Compute_Sim(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const int iteration,
                            const float dt, const float voxelSize, const cudaStream_t& stream) {
	Compute(data, handle, iteration, dt, voxelSize, stream);
}