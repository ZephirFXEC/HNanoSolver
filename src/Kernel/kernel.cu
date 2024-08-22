// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/tools/CreateNanoGrid.h>

__device__ nanovdb::Vec3f sampleVelocityGrid(const nanovdb::Vec3fGrid* velGrid,
                                             const nanovdb::Vec3f& pos) {
	// Get the tree from the velocity grid
	const auto& tree = velGrid->tree();

	// Convert world position to index space (voxel coordinates)
	const nanovdb::Vec3f voxelPos = (pos - velGrid->worldBBox().min()) / nanovdb::Vec3f(velGrid->voxelSize());

	// Trilinear interpolation
	const nanovdb::Coord lowerCoord = nanovdb::Coord::Floor(voxelPos);
	const nanovdb::Vec3f frac = voxelPos - lowerCoord.asVec3s();

	const nanovdb::Vec3f v000 = tree.getValue(lowerCoord);
	const nanovdb::Vec3f v100 = tree.getValue(lowerCoord.offsetBy(1, 0, 0));
	const nanovdb::Vec3f v010 = tree.getValue(lowerCoord.offsetBy(0, 1, 0));
	const nanovdb::Vec3f v001 = tree.getValue(lowerCoord.offsetBy(0, 0, 1));
	const nanovdb::Vec3f v110 = tree.getValue(lowerCoord.offsetBy(1, 1, 0));
	const nanovdb::Vec3f v101 = tree.getValue(lowerCoord.offsetBy(1, 0, 1));
	const nanovdb::Vec3f v011 = tree.getValue(lowerCoord.offsetBy(0, 1, 1));
	const nanovdb::Vec3f v111 = tree.getValue(lowerCoord.offsetBy(1, 1, 1));

	return (v000 * (1.0f - frac[0]) * (1.0f - frac[1]) * (1.0f - frac[2]) +
	        v100 * frac[0] * (1.0f - frac[1]) * (1.0f - frac[2]) +
	        v010 * (1.0f - frac[0]) * frac[1] * (1.0f - frac[2]) +
	        v001 * (1.0f - frac[0]) * (1.0f - frac[1]) * frac[2] + v101 * frac[0] * (1.0f - frac[1]) * frac[2] +
	        v011 * (1.0f - frac[0]) * frac[1] * frac[2] + v110 * frac[0] * frac[1] * (1.0f - frac[2]) +
	        v111 * frac[0] * frac[1] * frac[2]);
}

__device__ float interpolateDensity(nanovdb::FloatGrid* densGrid, const nanovdb::Vec3f& pos) {
	// Similar trilinear interpolation implementation for density grid
	const auto& tree = densGrid->tree();

	const nanovdb::Vec3f voxelPos = (pos - densGrid->worldBBox().min()) / nanovdb::Vec3f(densGrid->voxelSize());

	const nanovdb::Coord lowerCoord = nanovdb::Coord::Floor(voxelPos);
	const nanovdb::Vec3f frac = voxelPos - lowerCoord.asVec3s();

	const float d000 = tree.getValue(lowerCoord);
	const float d100 = tree.getValue(lowerCoord.offsetBy(1, 0, 0));
	const float d010 = tree.getValue(lowerCoord.offsetBy(0, 1, 0));
	const float d001 = tree.getValue(lowerCoord.offsetBy(0, 0, 1));
	const float d110 = tree.getValue(lowerCoord.offsetBy(1, 1, 0));
	const float d101 = tree.getValue(lowerCoord.offsetBy(1, 0, 1));
	const float d011 = tree.getValue(lowerCoord.offsetBy(0, 1, 1));
	const float d111 = tree.getValue(lowerCoord.offsetBy(1, 1, 1));

	return (d000 * (1.0f - frac[0]) * (1.0f - frac[1]) * (1.0f - frac[2]) +
	        d100 * frac[0] * (1.0f - frac[1]) * (1.0f - frac[2]) +
	        d010 * (1.0f - frac[0]) * frac[1] * (1.0f - frac[2]) +
	        d001 * (1.0f - frac[0]) * (1.0f - frac[1]) * frac[2] + d101 * frac[0] * (1.0f - frac[1]) * frac[2] +
	        d011 * (1.0f - frac[0]) * frac[1] * frac[2] + d110 * frac[0] * frac[1] * (1.0f - frac[2]) +
	        d111 * frac[0] * frac[1] * frac[2]);
}


__global__ void gpu_kernel(nanovdb::FloatGrid* deviceGrid, const nanovdb::Vec3fGrid* velGrid,
                           const uint64_t leafCount, const float dt) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= 512 * leafCount) return;

	auto& tree = deviceGrid->tree();

	auto* leaf_d = tree.getFirstNode<0>() + (idx >> 9);
	const int i = idx & 511;


	if (leaf_d->isActive(i)) {
		// Get the position of the voxel in index space
		const nanovdb::Coord voxelCoord = leaf_d->offsetToGlobalCoord(i);

		// Convert the voxel index to world space position
		const nanovdb::Vec3f voxelPosWorld =
		    deviceGrid->worldBBox().min() + voxelCoord.asVec3d() * deviceGrid->voxelSize();

		// Sample the velocity grid at the world space position
		const nanovdb::Vec3f velocity = sampleVelocityGrid(velGrid, voxelPosWorld);

		// Advect the density using the sampled velocity
		const nanovdb::Vec3f displacedPos = voxelPosWorld + velocity * dt;

		// Interpolate the density value at the displaced position
		const float advectedDensity = interpolateDensity(deviceGrid, displacedPos);

		// Set the new density value
		leaf_d->setValueOnly(i, advectedDensity);
	}
}

// This is called by the client code on the host
extern "C" void launch_kernels(nanovdb::FloatGrid* deviceGrid, const nanovdb::Vec3fGrid* velGrid,
                               const int leafCount, const float dt, cudaStream_t stream) {
	// Calculate the total number of voxels to process
	const uint64_t totalVoxels = 512 * leafCount;

	// Define block size and grid size
	int blockSize = 128;                                       // Number of threads per block
	int gridSize = (totalVoxels + blockSize - 1) / blockSize;  // Number of blocks

	// Launch the kernel
	gpu_kernel<<<gridSize, blockSize, 0, stream>>>(deviceGrid, velGrid, leafCount, dt);
}