#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

void Compute(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const int iteration,
             const float dt, const float voxelSize, const CombustionParams& params, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();
	const size_t numLeaves = totalVoxels / 512;
	const dim3 numBlocks(8, 8, 8);
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
	std::unordered_map<std::string, float*> d_inputs;
	std::unordered_map<std::string, float*> d_outputs;

	// Initialize resources for each float block
	for (size_t i = 0; i < floatBlocks.size(); i++) {
		cudaStreamCreate(&streams[i]);

		auto* host_ptr = data.pValues<float>(floatBlocks[i]);
		if (!host_ptr) {
			throw std::runtime_error("Block '" + floatBlocks[i] + "' not found or type mismatch");
		}
		hostPointers[i] = host_ptr;
		float* d_input;
		float* d_output;
		cudaMallocAsync(&d_input, totalVoxels * sizeof(float), streams[i]);
		cudaMallocAsync(&d_output, totalVoxels * sizeof(float), streams[i]);

		cudaMemcpyAsync(d_input, host_ptr, totalVoxels * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

		d_inputs[floatBlocks[i]] = d_input;
		d_outputs[floatBlocks[i]] = d_output;
	}

	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	nanovdb::Vec3f* d_outVel = nullptr;
	float* d_divergence = nullptr;
	float* d_pressure = nullptr;

	cudaMallocAsync(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), stream);
	cudaMallocAsync(&d_coords, totalVoxels * sizeof(nanovdb::Coord), stream);
	cudaMallocAsync(&d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), stream);
	cudaMallocAsync(&d_divergence, totalVoxels * sizeof(float), stream);
	cudaMallocAsync(&d_pressure, totalVoxels * sizeof(float), stream);

	cudaMemsetAsync(d_outVel, 0, totalVoxels * sizeof(nanovdb::Vec3f), stream);
	cudaMemsetAsync(d_divergence, 0, totalVoxels * sizeof(float), stream);
	cudaMemsetAsync(d_pressure, 0, totalVoxels * sizeof(float), stream);

	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);

	// Create grid and prepare for kernel launch
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	// Calculate optimal kernel launch parameters
	int blockSize = 256;
	const int gridSize = (totalVoxels + blockSize - 1) / blockSize;

	// Simulation pipeline with synchronization points
	// Step 1: Advect velocity field
	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Velocity", 12 * 2 /* Vec3f */ + 8 /* Coords */, totalVoxels);
		advect_vector<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_outVel, totalVoxels, dt, inv_voxelSize);
	}


	// Step 2: Apply buoyancy forces
	{
		ScopedTimerGPU timer("HNanoSolver::Combustion", 4 * 4 + 8 + 4 * 4 + 12 * 2 + 4, totalVoxels);

		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Combust", 4 * 4 /* float */, totalVoxels);

			combustion<<<gridSize, blockSize, 0, stream>>>(d_inputs["fuel"], d_inputs["temperature"], d_outputs["fuel"],
			                                               d_outputs["temperature"], dt, params.ignitionTemp, params.combustionRate,
			                                               params.heatRelease, totalVoxels);
		}
		std::swap(d_inputs["fuel"], d_outputs["fuel"]);
		std::swap(d_inputs["temperature"], d_outputs["temperature"]);

		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Diffusion", 8 + 4 * 4 /* float */, totalVoxels);

			diffusion<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_inputs["temperature"], d_inputs["fuel"],
			                                              d_outputs["temperature"], d_outputs["fuel"], dt, params.temperatureDiffusion,
			                                              params.fuelDiffusion, params.ambientTemp, totalVoxels);
		}

		std::swap(d_inputs["fuel"], d_outputs["fuel"]);
		std::swap(d_inputs["temperature"], d_outputs["temperature"]);

		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Buoyancy", 12 * 2 /* Vec3f */ + 4 /* float */, totalVoxels);
			temperature_buoyancy<<<gridSize, blockSize, 0, stream>>>(d_outVel, d_inputs["temperature"], d_velocity, dt, params.ambientTemp,
			                                                         params.buoyancyStrength, totalVoxels);
		}
	}

	// Step 3: Calculate velocity field divergence
	{
		ScopedTimerGPU timer("HNanoSolver::Divergence", 12 /* Vec3f */ + 8 + 4 /* float */, totalVoxels);
		divergence<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_divergence, inv_voxelSize, totalVoxels);
	}


	// Step 4: Pressure solver (Red-black Gauss-Seidel iterations)
	float omega = 2.0f / (1.0f + sin(3.14159f * voxelSize));
	{
		ScopedTimerGPU timer("HNanoSolver::Pressure", 8 + 4 * 2 /* float */, totalVoxels * iteration);

		for (int iter = 0; iter < iteration; ++iter) {
			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize,
			                                                              totalVoxels, 0, omega);

			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize,
			                                                              totalVoxels, 1, omega);
		}
	}


	// Step 5: Apply pressure gradient to enforce incompressibility
	{
		ScopedTimerGPU timer("HNanoSolver::Projection", 12 * 2 /* Vec3f */ + 8 + 4 /* float */, totalVoxels);
		subtractPressureGradient<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, totalVoxels, d_velocity, d_pressure, d_outVel,
		                                                             inv_voxelSize);
	}
	// Sync before advecting scalar fields
	cudaStreamSynchronize(stream);


	// Step 6: Advect all scalar fields in parallel using individual streams
	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Scalar", 12 /* Vec3f */ + 8 /* Coords */ + 4 * 2 /* float */, totalVoxels);
		for (size_t i = 0; i < floatBlocks.size(); ++i) {
			advect_scalar<<<gridSize, blockSize, 0, streams[i]>>>(gpuGrid, d_coords, d_outVel, d_inputs[floatBlocks[i]],
			                                                      d_outputs[floatBlocks[i]], totalVoxels, dt, inv_voxelSize);
		}
	}

	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		cudaMemcpyAsync(hostPointers[i], d_outputs[floatBlocks[i]], totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}

	// Synchronize all streams
	for (auto& s : streams) {
		cudaStreamSynchronize(s);
	}

	cudaStreamSynchronize(stream);

	cudaMemcpyAsync(velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, stream);

	// Clean up all allocated resources
	// Free device memory
	cudaFreeAsync(d_velocity, stream);
	cudaFreeAsync(d_coords, stream);
	cudaFreeAsync(d_outVel, stream);
	cudaFreeAsync(d_divergence, stream);
	cudaFreeAsync(d_pressure, stream);

	// Free scalar field resources
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		cudaFreeAsync(d_inputs[floatBlocks[i]], streams[i]);
		cudaFreeAsync(d_outputs[floatBlocks[i]], streams[i]);
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


extern "C" void Compute_Sim(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, int iteration,
                            float dt, float voxelSize, const CombustionParams& params, const cudaStream_t& stream) {
	Compute(data, handle, iteration, dt, voxelSize, params, stream);
}