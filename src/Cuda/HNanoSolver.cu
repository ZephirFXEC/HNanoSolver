#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

void Compute(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const int iteration,
             const float dt, const float voxelSize, const CombustionParams& params, const bool hasCollision, const cudaStream_t& stream) {
	// --- Input Validation ---
	if (voxelSize <= 0.0f) {
		throw std::invalid_argument("voxelSize must be positive.");
	}
	if (dt < 0.0f) {  // Allow dt == 0? Might mean no update step.
		throw std::invalid_argument("dt (time step) cannot be negative.");
	}
	if (iteration <= 0) {
		throw std::invalid_argument("Number of pressure iterations must be positive.");
	}
	if (handle.isEmpty()) {
		throw std::invalid_argument("Invalid nanovdb::GridHandle provided (null grid).");
	}

	const size_t totalVoxels = data.size();
	if (totalVoxels == 0) {
		return;  // Nothing to compute
	}

	// Get NanoVDB grid pointer (ensure it's the expected type)
	// Kernels expect nanovdb::NanoGrid<nanovdb::ValueOnIndex>*
	auto gpuGridPtr = handle.deviceGrid<nanovdb::ValueOnIndex>();
	if (!gpuGridPtr) {
		throw std::runtime_error("Failed to get device grid pointer of type ValueOnIndex from handle.");
	}

	const float inv_voxelSize = 1.0f / voxelSize;

	// --- 1. Get Host Data Pointers and Validate ---

	// Velocity
	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity), found " + std::to_string(vec3fBlocks.size()));
	}
	const std::string& velocityBlockName = vec3fBlocks[0];

	nanovdb::Vec3f* h_velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(velocityBlockName));
	if (!h_velocity) {
		throw std::runtime_error("Host velocity data pointer is null for block: " + velocityBlockName);
	}

	// Coordinates
	nanovdb::Coord* h_coords = reinterpret_cast<nanovdb::Coord*>(data.pCoords());
	if (!h_coords) {
		throw std::runtime_error("Host coordinate data pointer is null.");
	}

	// Scalar Fields (float)
	const auto floatBlockNames = data.getBlocksOfType<float>();
	if (floatBlockNames.empty()) {
		throw std::runtime_error("No float blocks found in input data.");
	}

	// Get collision SDF if available
	float* h_collisionSDF = nullptr;
	bool hasCollisionData = false;
	if (hasCollision) {
		if (std::find(floatBlockNames.begin(), floatBlockNames.end(), "collision_sdf") != floatBlockNames.end()) {
			h_collisionSDF = data.pValues<float>("collision_sdf");
			if (h_collisionSDF) {
				hasCollisionData = true;
			}
		}
	}

	std::vector<float*> h_floatPointers(floatBlockNames.size());
	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		h_floatPointers[i] = data.pValues<float>(floatBlockNames[i]);
		if (!h_floatPointers[i]) {
			throw std::runtime_error("Host float data pointer is null for block: " + floatBlockNames[i]);
		}
	}

	// --- 2. Allocate Device Memory using RAII ---

	DeviceMemory<nanovdb::Vec3f> d_velocity(totalVoxels, stream);     // Input velocity / Final output velocity
	DeviceMemory<nanovdb::Vec3f> d_advectedVel(totalVoxels, stream);  // Intermediate velocity after advection/buoyancy
	DeviceMemory<nanovdb::Coord> d_coords(totalVoxels, stream);
	DeviceMemory<float> d_divergence(totalVoxels, stream);
	DeviceMemory<float> d_pressure(totalVoxels, stream);

	// Allocate memory for SDF collision if available
	DeviceMemory<float> d_collisionSDF;
	if (hasCollisionData) {
		d_collisionSDF = DeviceMemory<float>(totalVoxels, stream);
	}

	std::unordered_map<std::string, DeviceMemory<float>> d_inputs;
	std::unordered_map<std::string, DeviceMemory<float>> d_outputs;

	for (const auto& name : floatBlockNames) {
		// Use emplace for direct construction within the map
		d_inputs.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(totalVoxels, stream));
		d_outputs.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(totalVoxels, stream));
	}

	// --- 3. Initialize Device Memory (Memset & Memcpy H->D) ---

	// Memset intermediate/output buffers
	CUDA_CHECK(cudaMemsetAsync(d_advectedVel.get(), 0, d_advectedVel.bytes(), stream));
	CUDA_CHECK(cudaMemsetAsync(d_divergence.get(), 0, d_divergence.bytes(), stream));
	CUDA_CHECK(cudaMemsetAsync(d_pressure.get(), 0, d_pressure.bytes(), stream));
	// Output scalars don't strictly need zeroing if fully overwritten, but doesn't hurt.
	for (auto& [fst, snd] : d_outputs) {
		CUDA_CHECK(cudaMemsetAsync(snd.get(), 0, snd.bytes(), stream));
	}

	// Copy initial data from Host to Device (Asynchronously)
	CUDA_CHECK(cudaMemcpyAsync(d_velocity.get(), h_velocity, d_velocity.bytes(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_coords.get(), h_coords, d_coords.bytes(), cudaMemcpyHostToDevice, stream));

	// Copy collision SDF if available
	if (hasCollisionData) {
		CUDA_CHECK(cudaMemcpyAsync(d_collisionSDF.get(), h_collisionSDF, d_collisionSDF.bytes(), cudaMemcpyHostToDevice, stream));
	}

	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		const std::string& name = floatBlockNames[i];
		float* h_ptr = h_floatPointers[i];
		auto& d_input_mem = d_inputs.at(name);  // Use .at() for checked access
		CUDA_CHECK(cudaMemcpyAsync(d_input_mem.get(), h_ptr, d_input_mem.bytes(), cudaMemcpyHostToDevice, stream));
	}


	// --- 4. Kernel Launch Configuration ---
	constexpr int PREFERRED_BLOCK_SIZE = 256;
	// Ensure block size is not larger than the maximum allowed or the number of elements
	// Note: Kernels might have internal assumptions about block size (e.g., shared memory).
	// If using _opt kernels, block size must match their requirements (e.g., 8x8x8=512).
	// For the current non-opt kernels, 256 is usually safe.
	int maxThreadsPerBlock = 0;
	maxThreadsPerBlock = 1024;  // Assume a reasonable default if not querying
	dim3 leafDim(8, 8, 8);
	uint32_t leafNum = totalVoxels / 512;

	const int blockSize = std::min((int)totalVoxels, std::min(PREFERRED_BLOCK_SIZE, maxThreadsPerBlock));
	const int gridSize = (totalVoxels + blockSize - 1) / blockSize;

	// --- 5. Simulation Pipeline Kernels ---

	// If collision is enabled, enforce initial collision boundaries
	if (hasCollisionData) {
		enforceCollisionBoundaries<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(), d_collisionSDF.get(),
		                                                               voxelSize, totalVoxels);
		CUDA_CHECK(cudaGetLastError());
	}

	// Step 1: Advect velocity field
	// Input: d_velocity (current state), d_coords
	// Output: d_advectedVel (intermediate buffer)
	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Velocity", 12 * 10 + 12, totalVoxels);
		advect_vector<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(),
		                                                  d_velocity.get(),     // Velocity field to advect (and sample from)
		                                                  d_advectedVel.get(),  // Output buffer for advected result
		                                                  hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, totalVoxels,
		                                                  dt, inv_voxelSize);
		CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
	}

	{
		ScopedTimerGPU timer("HNanoSolver::VorticityConfinement", 12 * 43, totalVoxels);
		vorticityConfinement<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_advectedVel.get(), d_advectedVel.get(), dt,
		                                                         inv_voxelSize, params.vorticityScale, params.factorScale, totalVoxels);
	}

	// Step 2: Calculate velocity field divergence
	// Input: d_velocity (using original velocity for divergence, common practice), d_coords
	// Output: d_divergence
	{
		ScopedTimerGPU timer("HNanoSolver::Divergence", 12 + 12 * 6 + 4, totalVoxels);
		// Using the non-optimized divergence kernel as in the original code
		divergence<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(),
		                                               d_velocity.get(),  // Use original velocity for divergence calc
		                                               d_divergence.get(), inv_voxelSize, totalVoxels);
		CUDA_CHECK(cudaGetLastError());
	}

	// Step 3: Combustion and Buoyancy
	{
		// Validate required fields exist before launching kernels
		const std::vector<std::string> required_comb_fields = {"fuel", "waste", "temperature", "flame"};
		for (const auto& field : required_comb_fields) {
			if (d_inputs.find(field) == d_inputs.end()) {
				throw std::runtime_error("Missing required input field for combustion: " + field);
			}
			if (d_outputs.find(field) == d_outputs.end()) {
				throw std::runtime_error("Missing required output field for combustion: " + field);
			}
		}
		// Check temperature exists for buoyancy
		if (d_outputs.find("temperature") == d_outputs.end()) {  // Buoyancy uses output T
			throw std::runtime_error("Missing output buffer for temperature (needed for buoyancy)");
		}


		// Launch Combustion Kernel
		// Input: d_inputs[...], d_divergence (read/write)
		// Output: d_outputs[...]
		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Combust", 4 * 9, totalVoxels);
			combustion_oxygen<<<gridSize, blockSize, 0, stream>>>(
			    d_inputs.at("fuel").get(), d_inputs.at("waste").get(), d_inputs.at("temperature").get(),
			    d_divergence.get(),  // Input/Output - combustion adds expansion
			    d_inputs.at("flame").get(),
			    d_outputs.at("fuel").get(),  // Write results to output buffers
			    d_outputs.at("waste").get(), d_outputs.at("temperature").get(), d_outputs.at("flame").get(), params.temperatureRelease,
			    params.expansionRate, totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		}

		// Launch Buoyancy Kernel
		// Input: d_velocity (original velocity to base force on), d_outputs["temperature"] (result from combustion)
		// Output: d_advectedVel (add buoyancy force to the already advected velocity)
		{
			ScopedTimerGPU t("HNanoSolver::Buoyancy", 12 * 2 + 4, totalVoxels);  // Estimate bytes
			temperature_buoyancy<<<gridSize, blockSize, 0, stream>>>(
			    d_advectedVel.get(),                // Read the intermediate (advected) velocity
			    d_outputs.at("temperature").get(),  // Read temperature *after* combustion
			    d_advectedVel.get(),                // Write buoyancy force additively back to the intermediate velocity
			    dt, params.ambientTemp, params.buoyancyStrength, totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		}

		// SWAP BUFFERS for fields modified by combustion
		// The results are in d_outputs. They need to be the input for the next step (scalar advection).
		// Move the DeviceMemory objects.
		for (const auto& field : required_comb_fields) {
			// Move the result from output map to input map
			d_inputs.at(field) = std::move(d_outputs.at(field));
			// Create a new (empty) output buffer in the output map for the next step
			d_outputs.at(field) = DeviceMemory<float>(totalVoxels, stream);
			// Need to memset the new output buffer if kernels don't guarantee writing all voxels
			CUDA_CHECK(cudaMemsetAsync(d_outputs.at(field).get(), 0, d_outputs.at(field).bytes(), stream));
		}
		// Fields not in required_comb_fields still have original data in d_inputs
		// and (potentially zeroed) buffers in d_outputs.

	}  // End Combustion/Buoyancy block


	// Step 4: Pressure solver (Red-black Gauss-Seidel iterations)
	// Input: d_divergence (potentially modified by combustion), d_coords
	// Output: d_pressure (updated iteratively)
	{
		const float omega = 2.0f / (1.0f + sinf(static_cast<float>(3.14159) * voxelSize));  // Using cmath
		ScopedTimerGPU timer("HNanoSolver::Pressure", 12 + 4 * 9, totalVoxels * iteration);

		for (int iter = 0; iter < iteration; ++iter) {
			// Red phase
			// Using the non-optimized redBlackGaussSeidelUpdate kernel
			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(), d_pressure.get(),
			                                                              voxelSize, totalVoxels, 0, omega);  // color = 0 for red

			// Black phase
			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(), d_pressure.get(),
			                                                              voxelSize, totalVoxels, 1, omega);  // color = 1 for black
		}

		CUDA_CHECK(cudaGetLastError());
	}


	// Step 5: Apply pressure gradient (Projection)
	// Input: d_advectedVel (intermediate velocity), d_pressure, d_coords
	// Output: d_velocity (final divergence-free velocity for this timestep)
	{
		constexpr int bytes_per_voxel = sizeof(nanovdb::Vec3f) * 2 + sizeof(float) * 6;  // 48
		ScopedTimerGPU timer("HNanoSolver::Projection", bytes_per_voxel, totalVoxels);
		// Using the non-optimized subtractPressureGradient kernel
		subtractPressureGradient<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), totalVoxels,
		                                                             d_advectedVel.get(),  // Input velocity field to correct (u*)
		                                                             d_pressure.get(),     // Pressure field
		                                                             d_velocity.get(),     // Output projected velocity (u_n+1)
		                                                             hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData,
		                                                             inv_voxelSize);
		CUDA_CHECK(cudaGetLastError());
	}

	// Apply collision boundaries again after projection
	if (hasCollisionData) {
		enforceCollisionBoundaries<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(), d_collisionSDF.get(),
		                                                               voxelSize, totalVoxels);
		CUDA_CHECK(cudaGetLastError());
	}

	// Step 6: Advect all scalar fields
	// Input: d_velocity (final projected velocity), d_inputs (state including post-combustion), d_coords
	// Output: d_outputs
	/*
	{
	    ScopedTimerGPU timer("HNanoSolver::Advect::Scalar", 12 * 2 + 12 + 4 * 10, totalVoxels);
	    for (const auto& name : floatBlockNames) {
	        // Ensure input/output buffers exist (should always do after setup/swap)
	        if (d_inputs.find(name) == d_inputs.end() || d_outputs.find(name) == d_outputs.end()) {
	            // This should ideally not happen if setup/swap logic is correct
	            throw std::logic_error("Internal error: Missing input/output buffer for scalar advection: " + name);
	        }

	        advect_scalar<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(),
	                                                          d_velocity.get(),          // Advect using the final projected velocity
	                                                          d_inputs.at(name).get(),   // Scalar field state before advection
	                                                          d_outputs.at(name).get(),  // Write advected result here
	                                                          hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData,
	                                                          totalVoxels, dt, inv_voxelSize);
	    }
	    CUDA_CHECK(cudaGetLastError());  // Check each kernel launch
	}
	*/
	{
		std::vector<float*> d_inDataArrays;
		std::vector<float*> d_outDataArrays;

		// Populate arrays from your maps
		for (const auto& name : floatBlockNames) {
			if (name == "collision_sdf") continue;
			d_inDataArrays.push_back(d_inputs.at(name).get());
			d_outDataArrays.push_back(d_outputs.at(name).get());
		}

		// Allocate device-side array pointers
		float** d_inDataArraysDev;
		float** d_outDataArraysDev;

		cudaMalloc(&d_inDataArraysDev, d_inDataArrays.size() * sizeof(float*));
		cudaMalloc(&d_outDataArraysDev, d_outDataArrays.size() * sizeof(float*));

		cudaMemcpyAsync(d_inDataArraysDev, d_inDataArrays.data(), d_inDataArrays.size() * sizeof(float*), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_outDataArraysDev, d_outDataArrays.data(), d_outDataArrays.size() * sizeof(float*), cudaMemcpyHostToDevice,
		                stream);

		ScopedTimerGPU timer("HNanoSolver::Advect::Scalar", 12 * 2 + 12 + 4 * 10, totalVoxels);

		// Launch the single kernel pass
		advect_scalars<<<gridSize, blockSize, 0, stream>>>(
		    gpuGridPtr, d_coords.get(), d_velocity.get(), d_inDataArraysDev, d_outDataArraysDev, static_cast<int>(d_inDataArrays.size()),
		    hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, totalVoxels, dt, inv_voxelSize);

		// Free temporary arrays
		cudaFree(d_inDataArraysDev);
		cudaFree(d_outDataArraysDev);


		CUDA_CHECK(cudaGetLastError());  // Check each kernel launch
	}

	// --- 6. Copy Results back from Device to Host ---

	// Copy final projected velocity
	CUDA_CHECK(cudaMemcpyAsync(h_velocity, d_velocity.get(), d_velocity.bytes(), cudaMemcpyDeviceToHost, stream));

	// Copy final advected scalar fields (results are in d_outputs after last step)
	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		const std::string& name = floatBlockNames[i];
		float* h_ptr = h_floatPointers[i];        // Get corresponding host pointer
		auto& d_output_mem = d_outputs.at(name);  // Result is in output buffer from scalar advection
		CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_output_mem.get(), d_output_mem.bytes(), cudaMemcpyDeviceToHost, stream));
	}

	cudaStreamSynchronize(stream);
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
                            float dt, float voxelSize, const CombustionParams& params, bool hasCollision, const cudaStream_t& stream) {
	Compute(data, handle, iteration, dt, voxelSize, params, hasCollision, stream);
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
        redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0,
0, 1.0f); redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0,
1, 1.0f);
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
        redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, dx1,
totalVoxels1, 0, 1.0f); redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1,
dx1, totalVoxels1, 1, 1.0f);
    }

    // Compute residual on Level 1: residual1 = divergence1 - A * pressure1
    compute_residual<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, residual1, dx1,
totalVoxels1);

    // --- Restrict from Level 1 (4^3) to Level 2 (2^3) ---
    // Use a kernel that restricts a 4^3 block (64 voxels) to a 2^3 block (8 voxels)
    int blockSizeR1 = 256;
    int gridSize2 = (totalVoxels2 + blockSizeR1 - 1) / blockSizeR1;
    restrict_to_2x2x2<<<gridSize2, blockSizeR1, 0, stream>>>(domainGrid, d_coord, residual1, divergence2, totalVoxels2);

    // Initialize Level 2 pressure to zero.
    cudaMemsetAsync(pressure2, 0, totalVoxels2 * sizeof(float), stream);

    // --- Level 2: Coarsest grid (2^3) ---
    for (int iter = 0; iter < coarseSmooth; ++iter) {
        redBlackGaussSeidelUpdate<<<gridSize2, blockSize, 0, stream>>>(domainGrid, d_coord, divergence2, pressure2, dx2, totalVoxels2,
0, 1.0f); redBlackGaussSeidelUpdate<<<gridSize2, blockSize, 0, stream>>>(domainGrid, d_coord, divergence2, pressure2, dx2, totalVoxels2,
1, 1.0f);
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
        redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1, dx1,
totalVoxels1, 0, 1.0f); redBlackGaussSeidelUpdate<<<gridSizeLevel1, blockSize, 0, stream>>>(domainGrid, d_coord, divergence1, pressure1,
dx1, totalVoxels1, 1, 1.0f);
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
        redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0,
0, 1.0f); redBlackGaussSeidelUpdate<<<gridSize0, blockSize, 0, stream>>>(domainGrid, d_coord, divergence0, pressure0, dx0, totalVoxels0,
1, 1.0f);
    }
}

*/