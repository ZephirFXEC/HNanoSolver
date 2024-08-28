#include <cuda/std/__algorithm/clamp.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cuda/std/cmath>

template <typename T>
__device__ inline T lerp(T v0, T v1, T t) {
	return fma(t, v1, fma(-t, v0, v0));
}

__global__ void gpu_kernel(nanovdb::FloatGrid* deviceGrid, const nanovdb::Vec3fGrid* velGrid, const uint64_t leafCount,
                           const float voxelSize, const float dt) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= 512 * leafCount) return;

	auto& dtree = deviceGrid->tree();
	auto& vtree = velGrid->tree();

	auto* leaf_d = dtree.getFirstNode<0>() + (idx >> 9);
	const int i_d = idx & 511;

	auto* leaf_v = vtree.getFirstNode<0>() + (idx >> 9);

	const auto velAccessor = velGrid->getAccessor();
	const auto denAccessor = deviceGrid->getAccessor();
	const auto velSampler = nanovdb::createSampler<1>(velAccessor);
	const auto denSampler = nanovdb::createSampler<1>(denAccessor);

	if (leaf_v->isActive()) {
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

		// Set the new density value
		leaf_d->setValue(voxelCoord, new_density);
	}
}

// This is called by the client code on the host
extern "C" void kernel(nanovdb::FloatGrid* deviceGrid, const nanovdb::Vec3fGrid* velGrid, const int leafCount,
                               const float voxelSize, const float dt, cudaStream_t stream) {
	// Calculate the total number of voxels to process
	const uint64_t totalVoxels = 512 * leafCount;

	// Define block size and grid size
	int blockSize = 1024;                                      // Number of threads per block
	int gridSize = (totalVoxels + blockSize - 1) / blockSize;  // Number of blocks

	// Launch the kernel
	gpu_kernel<<<gridSize, blockSize, 0, stream>>>(deviceGrid, velGrid, leafCount, voxelSize, dt);
}