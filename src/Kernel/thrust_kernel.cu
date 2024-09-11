#include <cuda/std/__algorithm/clamp.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cuda/std/cmath>

#include "utils.cuh"


extern "C" void vel_thrust_kernel(const nanovdb::Vec3fGrid* velGrid, const uint64_t leafCount, const float voxelSize,
                                  const float dt, cudaStream_t stream, nanovdb::Coord* h_coords,
                                  nanovdb::Vec3f* h_values, size_t& count) {
	size_t* voxelCount = nullptr;
	cudaCheck(cudaMalloc(&voxelCount, sizeof(size_t)));
	cudaCheck(cudaMemset(voxelCount, 0, sizeof(size_t)));

	constexpr unsigned int numThreads = 256;
	const unsigned int numVoxels = 512 * leafCount;
	const unsigned int numBlocks = blocksPerGrid(numVoxels, numThreads);

	nanovdb::Coord* d_coords = nullptr;
	nanovdb::Vec3f* d_values = nullptr;

	cudaCheck(cudaMalloc(&d_coords, numVoxels * sizeof(nanovdb::Coord)));
	cudaCheck(cudaMalloc(&d_values, numVoxels * sizeof(nanovdb::Vec3f)));
	cudaCheck(cudaMemset(d_coords, 0, numVoxels * sizeof(nanovdb::Coord)));
	cudaCheck(cudaMemset(d_values, 0, numVoxels * sizeof(nanovdb::Vec3f)));

	cudaDeviceSynchronize();

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(numVoxels, [velGrid, voxelSize, dt, voxelCount, d_coords,
	                                                               d_values] __device__(const uint64_t n) {
		const auto& vtree = velGrid->tree();
		const int i_d = n & 511;
		const auto* leaf_v = vtree.getFirstNode<0>() + (n >> 9);
		const auto velAccessor = velGrid->getAccessor();
		const auto velSampler = nanovdb::createSampler<1>(velAccessor);

		if (leaf_v->isActive()) {
			// Get the position of the voxel in index space
			const nanovdb::Coord voxelCoord = leaf_v->offsetToGlobalCoord(i_d);
			const nanovdb::Vec3f voxelCoordf = voxelCoord.asVec3s();
			const nanovdb::Vec3f velocity = velSampler(voxelCoordf);

			// Forward step
			const nanovdb::Vec3f forward_pos = voxelCoordf - velocity * (dt / voxelSize);
			const nanovdb::Vec3f v_forward = velSampler(forward_pos);

			// Backward step
			const nanovdb::Vec3f back_pos = voxelCoordf + velSampler(forward_pos) * (dt / voxelSize);
			const nanovdb::Vec3f v_backward = velSampler(back_pos);

			// Error estimation and correction
			const nanovdb::Vec3f error = 0.5f * (velocity - v_backward);
			nanovdb::Vec3f v_corrected = v_forward + error;

			// Limit the correction based on the neighborhood of the forward position
			const auto max_correction = nanovdb::Vec3f(cuda::std::abs(0.5f * (v_forward[0] - velocity[0])),
			                                           cuda::std::abs(0.5f * (v_forward[1] - velocity[1])),
			                                           cuda::std::abs(0.5f * (v_forward[2] - velocity[2])));
			v_corrected[0] =
			    cuda::std::clamp(v_corrected[0], v_forward[0] - max_correction[0], v_forward[0] + max_correction[0]);
			v_corrected[1] =
			    cuda::std::clamp(v_corrected[1], v_forward[1] - max_correction[1], v_forward[1] + max_correction[1]);
			v_corrected[2] =
			    cuda::std::clamp(v_corrected[2], v_forward[2] - max_correction[2], v_forward[2] + max_correction[2]);

			// Final advection (blend between semi-Lagrangian and BFECC result)
			constexpr float blend_factor = 0.8f;  // Adjust this value between 0 and 1
			nanovdb::Vec3f new_velocity;
			new_velocity[0] = lerp(v_forward[0], v_corrected[0], blend_factor);
			new_velocity[1] = lerp(v_forward[1], v_corrected[1], blend_factor);
			new_velocity[2] = lerp(v_forward[2], v_corrected[2], blend_factor);

			const size_t index = atomicAdd(voxelCount, 1);
			d_coords[index] = voxelCoord;
			d_values[index] = new_velocity;
		}
	});

	cudaDeviceSynchronize();

	// Download the count of valid voxels
	size_t h_count;
	cudaCheck(cudaMemcpy(&h_count, voxelCount, sizeof(size_t), cudaMemcpyDeviceToHost));

	// Now you know how many valid voxels were processed and can download the data
	cudaCheck(cudaMemcpy(h_coords, d_coords, h_count * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(h_values, d_values, h_count * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost));

	count = h_count;

	// Free allocated memory
	cudaCheck(cudaFree(voxelCount));
	cudaCheck(cudaFree(d_coords));
	cudaCheck(cudaFree(d_values));
}

extern "C" void thrust_kernel(const nanovdb::FloatGrid* deviceGrid, const nanovdb::Vec3fGrid* velGrid,
                              const size_t leafCount, const float voxelSize, const float dt, cudaStream_t stream,
                              nanovdb::Coord* h_coords, float* h_values, size_t& count) {
	size_t* voxelCount = nullptr;
	cudaCheck(cudaMalloc(&voxelCount, sizeof(size_t)));
	cudaCheck(cudaMemset(voxelCount, 0, sizeof(size_t)));

	constexpr unsigned int numThreads = 256;
	const unsigned int numVoxels = 512 * leafCount;
	const unsigned int numBlocks = blocksPerGrid(numVoxels, numThreads);

	nanovdb::Coord* d_coords = nullptr;
	float* d_values = nullptr;

	cudaCheck(cudaMalloc(&d_coords, numVoxels * sizeof(nanovdb::Coord)));
	cudaCheck(cudaMalloc(&d_values, numVoxels * sizeof(float)));
	cudaCheck(cudaMemset(d_coords, 0, numVoxels * sizeof(nanovdb::Coord)));
	cudaCheck(cudaMemset(d_values, 0, numVoxels * sizeof(float)));

	cudaDeviceSynchronize();

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(
	    numVoxels, [deviceGrid, velGrid, voxelSize, dt, voxelCount, d_coords, d_values] __device__(const size_t n) {
		    const auto& dtree = deviceGrid->tree();

		    const auto* leaf_d = dtree.getFirstNode<0>() + (n >> 9);

		    const int i_d = n & 511;
		    const auto velAccessor = velGrid->getAccessor();
		    const auto denAccessor = deviceGrid->getAccessor();
		    const auto velSampler = nanovdb::createSampler<1>(velAccessor);
		    const auto denSampler = nanovdb::createSampler<1>(denAccessor);

		    if (leaf_d->isActive()) {
			    // Get the position of the voxel in index space
			    const nanovdb::Coord voxelCoord = leaf_d->offsetToGlobalCoord(i_d);
			    const nanovdb::Vec3f voxelCoordf = voxelCoord.asVec3s();
			    const float density = denSampler(voxelCoordf);

			    // Forward step
			    const nanovdb::Vec3f forward_pos = voxelCoordf - velSampler(voxelCoordf) * (dt / voxelSize);
			    const float d_forward = denSampler(forward_pos);

			    // Backward step
			    const nanovdb::Vec3f back_pos = voxelCoordf + velSampler(forward_pos) * (dt / voxelSize);
			    const float d_backward = denSampler(back_pos);

			    // Error estimation and correction
			    const float error = 0.5f * (density - d_backward);
			    float d_corrected = d_forward + error;

			    // Limit the correction based on the neighborhood of the forward position
			    const float max_correction = 0.5f * cuda::std::fabs(d_forward - density);
			    d_corrected = cuda::std::clamp(d_corrected, d_forward - max_correction, d_forward + max_correction);

			    // Final advection (blend between semi-Lagrangian and BFECC result)
			    constexpr float blend_factor = 0.8f;  // Adjust this value between 0 and 1
			    float new_density = lerp(d_forward, d_corrected, blend_factor);

			    // Ensure non-negativity
			    new_density = cuda::std::fmax(0.0f, new_density);

			    const size_t index = atomicAdd(voxelCount, 1);
			    d_coords[index] = voxelCoord;
			    d_values[index] = new_density;
		    }
	    });
	cudaCheckError();

	cudaDeviceSynchronize();

	// Download the count of valid voxels
	size_t h_count;
	cudaCheck(cudaMemcpy(&h_count, voxelCount, sizeof(size_t), cudaMemcpyDeviceToHost));

	// Now you know how many valid voxels were processed and can download the data
	cudaCheck(cudaMemcpy(h_coords, d_coords, h_count * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(h_values, d_values, h_count * sizeof(float), cudaMemcpyDeviceToHost));

	count = h_count;

	// Free allocated memory
	cudaCheck(cudaFree(voxelCount));
	cudaCheck(cudaFree(d_coords));
	cudaCheck(cudaFree(d_values));
}
