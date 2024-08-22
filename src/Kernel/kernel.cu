#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/tools/CreateNanoGrid.h>

__global__ void gpu_kernel(nanovdb::FloatGrid* deviceGrid, const nanovdb::Vec3fGrid* velGrid, const uint64_t leafCount,
                           const float dt) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= 512 * leafCount) return;

	auto& tree = deviceGrid->tree();

	auto* leaf_d = tree.getFirstNode<0>() + (idx >> 9);
	const int i = idx & 511;

	const auto velAccessor = velGrid->getAccessor();
	const auto denAccessor = tree.getAccessor();
	const auto velSampler = nanovdb::math::createSampler<1>(velAccessor);
	const auto denSampler = nanovdb::math::createSampler<1>(denAccessor);

	if (leaf_d->isActive(i)) {
		// Get the position of the voxel in index space
		const nanovdb::Coord voxelCoord = leaf_d->offsetToGlobalCoord(i);

		// Sample the velocity grid at the world space position
		const nanovdb::Vec3f velocity = velSampler(voxelCoord);

		// Advect the density using the sampled velocity
		const nanovdb::Vec3f displacedPos = voxelCoord.asVec3s() - velocity * dt;

		// Interpolate the density value at the displaced position
		const float advectedDensity = denSampler(displacedPos);

		// Set the new density value
		leaf_d->setValueOnly(i, advectedDensity);
	}
}

// This is called by the client code on the host
extern "C" void launch_kernels(nanovdb::FloatGrid* deviceGrid, const nanovdb::Vec3fGrid* velGrid, const int leafCount,
                               const float dt, cudaStream_t stream) {
	// Calculate the total number of voxels to process
	const uint64_t totalVoxels = 512 * leafCount;

	// Define block size and grid size
	int blockSize = 256;                                       // Number of threads per block
	int gridSize = (totalVoxels + blockSize - 1) / blockSize;  // Number of blocks

	// Launch the kernel
	gpu_kernel<<<gridSize, blockSize, 0, stream>>>(deviceGrid, velGrid, leafCount, dt);
}