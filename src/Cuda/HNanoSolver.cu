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
	int blockSize = 512;
	const int gridSize = totalVoxels / blockSize;

	// Simulation pipeline with synchronization points
	// Step 1: Advect velocity field
	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Velocity", 12 * 10 /* Vec3f */ + 12 /* Coords */, totalVoxels);
		advect_vector<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_outVel, totalVoxels, dt, inv_voxelSize);
	}


	// Step 2: Apply buoyancy forces
	{
		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Combust", 4 * 4 /* float */, totalVoxels);

			combustion<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_inputs["fuel"], d_inputs["temperature"], d_outputs["fuel"],
			                                               d_outputs["temperature"], dt, params.ignitionTemp, params.combustionRate,
			                                               params.heatRelease, totalVoxels);
		}

		std::swap(d_inputs["temperature"], d_outputs["temperature"]);
		std::swap(d_inputs["fuel"], d_outputs["fuel"]);

		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Diffusion", 12 + 4 * 16 /* float */, totalVoxels);

			diffusion<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_inputs["temperature"], d_inputs["fuel"],
			                                              d_outputs["temperature"], d_outputs["fuel"], dt, params.temperatureDiffusion,
			                                              params.fuelDiffusion, params.ambientTemp, totalVoxels);
		}

		std::swap(d_inputs["temperature"], d_outputs["temperature"]);
		std::swap(d_inputs["fuel"], d_outputs["fuel"]);

		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Buoyancy", 12 * 2 /* Vec3f */ + 4 /* float */, totalVoxels);
			temperature_buoyancy<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_outVel, d_inputs["temperature"], d_velocity, dt,
			                                                         params.ambientTemp, params.buoyancyStrength, totalVoxels);
		}
	}

	// Step 3: Calculate velocity field divergence
	{
		ScopedTimerGPU timer("HNanoSolver::Divergence", 12 /* Vec3f */ + 12 * 6 + 4 /* float */, totalVoxels);
		divergence<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_divergence, inv_voxelSize, totalVoxels);
	}


	// Step 4: Pressure solver (Red-black Gauss-Seidel iterations)
	float omega = 2.0f / (1.0f + sin(3.14159f * voxelSize));
	{
		ScopedTimerGPU timer("HNanoSolver::Pressure", 12 + 4 * 9 /* float */, totalVoxels * iteration);

		for (int iter = 0; iter < iteration; ++iter) {
			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize,
			                                                              totalVoxels, 0, omega);

			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize,
			                                                              totalVoxels, 1, omega);
		}
	}


	// Step 5: Apply pressure gradient to enforce incompressibility
	{
		constexpr int bytes_per_voxel = sizeof(nanovdb::Vec3f) * 2 + sizeof(float) * 6;  // 12*2 + 4*6 = 48
		ScopedTimerGPU timer("HNanoSolver::Projection", bytes_per_voxel, totalVoxels);
		// ScopedTimerGPU timer("HNanoSolver::Projection", 12 * 2 /* Vec3f */ + 4 /* float */, totalVoxels);
		subtractPressureGradient_opt<<<numLeaves, numBlocks, 0, stream>>>(gpuGrid, d_velocity, d_pressure, d_outVel, inv_voxelSize,
		                                                                  numLeaves);
	}

	// Sync before advecting scalar fields
	cudaStreamSynchronize(stream);


	// Step 6: Advect all scalar fields in parallel using individual streams
	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Scalar", 12 * 2 /* Vec3f */ + 12 /* Coords */ + 4 * 10 /* float */, totalVoxels);
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


/* Example host function for a three-level V-cycle
void v_cycle(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coord,
             // Level 0 (fine) arrays
             float* pressure0, float* divergence0, float* residual0,
             // Level 1 arrays (4^3 per leaf)
             float* pressure1, float* divergence1, float* residual1,
             // Level 2 arrays (coarsest: 2^3 per leaf)
             float* pressure2, float* divergence2, float* residual2,
             // Temporary correction arrays for prolongation
             float* correction1, float* correction0,
             // Grid sizes (total voxels) at each level
             size_t totalVoxels0, size_t totalVoxels1, size_t totalVoxels2,
             // Grid spacing at fine level (others are scaled accordingly)
             const float dx0, cudaStream_t stream) {
    // Smoothing parameters
    const int preSmooth = 3;
    const int postSmooth = 3;
    const int coarseSmooth = 10;  // for the small 2^3 system; you might use a direct solver here

    // Compute dx at each level
    const float dx1 = dx0 * 2.0f;
    const float dx2 = dx0 * 4.0f;

    // --- Level 0: Fine grid (8^3 per leaf) ---
    int blockSize = 256;
    int gridSize0 = (totalVoxels0 + blockSize - 1) / blockSize;

    // Pre-smoothing on fine level using red-black Gauss–Seidel (alternating colors)
    for (int iter = 0; iter < preSmooth; ++iter) {
        redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0, 0,
                                                                       1.0f);
        redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0, 1,
                                                                       1.0f);
    }

    // Compute residual at Level 0: residual0 = divergence0 - A * pressure0
    compute_residual<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, residual0, dx0, totalVoxels0);

    // --- Restrict from Level 0 (8^3) to Level 1 (4^3) ---
    // totalVoxels1 should equal (number_of_leaves * 64)
    int blockSizeR0 = 256;
    int gridSize1 = (totalVoxels1 + blockSizeR0 - 1) / blockSizeR0;
    restrict_to_4x4x4<<<gridSize1, blockSizeR0, 0, stream>>>(domainGrid, d_coord, residual0, divergence1, totalVoxels1);

    // Initialize Level 1 pressure to zero.
    cudaMemsetAsync(pressure1, 0, totalVoxels1 * sizeof(float), stream);

    // --- Level 1: Intermediate grid (4^3) ---
    // Pre-smoothing on Level 1
    int gridSizeLevel1 = gridSize1;  // already computed from totalVoxels1
    for (int iter = 0; iter < preSmooth; ++iter) {
        redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, dx1, totalVoxels1,
                                                                            0, 1.0f);
        redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, dx1, totalVoxels1,
                                                                            1, 1.0f);
    }

    // Compute residual on Level 1: residual1 = divergence1 - A * pressure1
    compute_residual<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, residual1, dx1, totalVoxels1);

    // --- Restrict from Level 1 (4^3) to Level 2 (2^3) ---
    // Use a kernel that restricts a 4^3 block (64 voxels) to a 2^3 block (8 voxels)
    int blockSizeR1 = 256;
    int gridSize2 = (totalVoxels2 + blockSizeR1 - 1) / blockSizeR1;
    restrict_to_2x2x2<<<gridSize2, blockSizeR1, 0, stream>>>(domainGrid, d_coord, residual1, divergence2, totalVoxels2);

    // Initialize Level 2 pressure to zero.
    cudaMemsetAsync(pressure2, 0, totalVoxels2 * sizeof(float), stream);

    // --- Level 2: Coarsest grid (2^3) ---
    for (int iter = 0; iter < coarseSmooth; ++iter) {
        redBlackGaussSeidelUpdate<<<gridSize2, blockSize, 0, stream>>>(domainGrid, d_coord, divergence2, pressure2, dx2, totalVoxels2, 0,
                                                                       1.0f);
        redBlackGaussSeidelUpdate<<<gridSize2, blockSize, 0, stream>>>(domainGrid, d_coord, divergence2, pressure2, dx2, totalVoxels2, 1,
                                                                       1.0f);
    }

    // --- Prolongate from Level 2 (2^3) to Level 1 (4^3) ---
    // Here, prolongateKernel is assumed to map an 2^3 block to a 4^3 block.
    // gridSize2 can be reused if the kernel is launched per coarse block.
    prolongate<<<gridSize2, blockSize, 0, stream>>>(pressure2, correction1, nanovdb::Coord(2, 2, 2), nanovdb::Coord(4, 4, 4)
                                                    /* coarse dimensions for Level 2 (e.g., 2,2,2)
                                                    /* fine dimensions for Level 1 (e.g., 4,4,4) );

    // Update Level 1 pressure: p1 = p1 + correction1
    update_pressure<<<gridSizeLevel1, blockSize, 0, stream>>>(totalVoxels1, pressure1, correction1);

    // Post-smoothing on Level 1
    for (int iter = 0; iter < postSmooth; ++iter) {
        redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, dx1, totalVoxels1,
                                                                            0, 1.0f);
        redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, dx1, totalVoxels1,
                                                                            1, 1.0f);
    }

    // --- Prolongate from Level 1 (4^3) to Level 0 (8^3) ---
    // Prolongate from the intermediate grid back to the fine grid.
    prolongate<<<gridSizeLevel1, blockSize, 0, stream>>>(pressure1, correction0, nanovdb::Coord(4, 4, 4), nanovdb::Coord(8, 8, 8)
                                                         /* coarse dimensions for Level 1 (e.g., 4,4,4)
                                                         /* fine dimensions for Level 0 (e.g., 8,8,8) );

    // Update Level 0 pressure: p0 = p0 + correction0
    update_pressure<<<gridSize0, blockSize, 0, stream>>>(totalVoxels0, pressure0, correction0);

    // --- Post-smoothing on fine grid (Level 0) ---
    for (int iter = 0; iter < postSmooth; ++iter) {
        redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0, 0,
                                                                       1.0f);
        redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0, 1,
                                                                       1.0f);
    }
}

*/

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