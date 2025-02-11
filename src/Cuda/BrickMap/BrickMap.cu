#include "BrickMap.cuh"


__host__ void BrickMap::initialize() {
	CHECK_CUDA(cudaMalloc(&d_brickMap, TOTAL_BRICKS_IN_MAP * sizeof(int)));
	// Use cudaMemset to initialize all entries to -1 (0xFFFFFFFF)
	CHECK_CUDA(cudaMemset(d_brickMap, 0xFF, TOTAL_BRICKS_IN_MAP * sizeof(int)));

	CHECK_CUDA(cudaMalloc(&d_bricks, _maxBricks * sizeof(Brick)));
	CHECK_CUDA(cudaMalloc(&d_freeList, _maxBricks * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_freeListTop, sizeof(int)));

	int threads = 256;
	int blocks = (_maxBricks + threads - 1) / threads;
	initFreeListKernel<<<blocks, threads>>>(d_freeList, d_freeListTop, _maxBricks);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
}

__host__ void BrickMap::buildFromUpdates(const std::vector<VoxelUpdate>& updates) const {
	int numUpdates = static_cast<int>(updates.size());
	VoxelUpdate* d_updates = nullptr;
	CHECK_CUDA(cudaMalloc(&d_updates, numUpdates * sizeof(VoxelUpdate)));
	CHECK_CUDA(cudaMemcpy(d_updates, updates.data(), numUpdates * sizeof(VoxelUpdate), cudaMemcpyHostToDevice));

	int threads = 256;
	int blocks = (numUpdates + threads - 1) / threads;
	buildBrickMapKernel<<<blocks, threads>>>(d_updates, numUpdates, d_brickMap, d_bricks, d_freeList, d_freeListTop);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaFree(d_updates));
}


__host__ void BrickMap::updateVoxel(int x, int y, int z, const Voxel& newVoxel) const {
	updateVoxelKernelClass<<<1, 1>>>(*this, x, y, z, newVoxel);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
}

__host__ Voxel BrickMap::queryVoxel(int x, int y, int z) const {
	int brick_x = x / BRICK_DIM;
	int brick_y = y / BRICK_DIM;
	int brick_z = z / BRICK_DIM;

	// 2. Check global boundaries
	if (brick_x < 0 || brick_x >= BRICK_MAP_DIM ||
		brick_y < 0 || brick_y >= BRICK_MAP_DIM ||
		brick_z < 0 || brick_z >= BRICK_MAP_DIM) {
		return Voxel();  // Return empty voxel for out of bounds
		}

	// 3. Calculate grid index
	int grid_index = brick_x + BRICK_MAP_DIM * (brick_y + BRICK_MAP_DIM * brick_z);

	// 4. Safely read brick index
	int brickIndex = -1;  // Initialize to invalid
	cudaMemcpy(&brickIndex, d_brickMap + grid_index, sizeof(int), cudaMemcpyDeviceToHost);
	if (brickIndex < 0) {
		return Voxel();  // Return empty voxel for missing brick or error
	}

	// 5. Calculate local coordinates within brick
	int local_x = x % BRICK_DIM;
	int local_y = y % BRICK_DIM;
	int local_z = z % BRICK_DIM;
	int voxelIndex = local_x + BRICK_DIM * (local_y + BRICK_DIM * local_z);

	// 6. Validate voxel index
	if (voxelIndex < 0 || voxelIndex >= VOXELS_PER_BRICK) {
		return Voxel();  // Return empty voxel for invalid local coordinates
	}

	Voxel value;
	cudaMemcpy(&value, &(d_bricks[brickIndex].voxels[voxelIndex]), sizeof(Voxel), cudaMemcpyDeviceToHost);

	return value;
}


__host__ void BrickMap::cleanupEmptyBricks() const {
	int threads = 256;
	int blocks = (_maxBricks + threads - 1) / threads;
	cleanupEmptyBricksKernel<<<blocks, threads>>>(d_brickMap, d_bricks, d_freeList, d_freeListTop, _maxBricks);
	CHECK_CUDA(cudaGetLastError());
}


__device__ void BrickMap::deviceUpdateVoxel(int x, int y, int z, const Voxel& newVoxel) const {
	int brick_x = x / BRICK_DIM;
	int brick_y = y / BRICK_DIM;
	int brick_z = z / BRICK_DIM;
	if (brick_x < 0 || brick_x >= BRICK_MAP_DIM || brick_y < 0 || brick_y >= BRICK_MAP_DIM || brick_z < 0 || brick_z >= BRICK_MAP_DIM)
		return;

	int grid_index = brick_x + BRICK_MAP_DIM * (brick_y + BRICK_MAP_DIM * brick_z);

	// Reserve a brick if needed.
	int brickIndex = atomicCAS(&d_brickMap[grid_index], -1, -2);
	if (brickIndex == -1) {
		int freeCount = atomicSub(d_freeListTop, 1);
		if (freeCount <= 0) {
			atomicAdd(d_freeListTop, 1);
			return;  // Out-of-memory.
		}
		int newBrickIndex = d_freeList[freeCount - 1];
		Brick& brick = d_bricks[newBrickIndex];
		for (int i = 0; i < VOXELS_PER_BRICK; i++) brick.voxels[i] = Voxel();
		brick.occupancy = 0;
		atomicExch(&d_brickMap[grid_index], newBrickIndex);
		brickIndex = newBrickIndex;
	} else {
		while (brickIndex == -2) brickIndex = d_brickMap[grid_index];
	}
	Brick* brick = &d_bricks[brickIndex];
	int local_x = x % BRICK_DIM;
	int local_y = y % BRICK_DIM;
	int local_z = z % BRICK_DIM;
	int voxelIndex = local_x + BRICK_DIM * (local_y + BRICK_DIM * local_z);
	Voxel oldVoxel = brick->voxels[voxelIndex];
	brick->voxels[voxelIndex] = newVoxel;
	bool wasEmpty = Voxel::isEmpty(oldVoxel);
	bool isNowEmpty = Voxel::isEmpty(newVoxel);
	if (wasEmpty && !isNowEmpty)
		atomicAdd(&brick->occupancy, 1);
	else if (!wasEmpty && isNowEmpty)
		atomicSub(&brick->occupancy, 1);
}

__device__ void BrickMap::deviceQueryVoxel(int x, int y, int z, Voxel& result) const {
	int brick_x = x / BRICK_DIM;
	int brick_y = y / BRICK_DIM;
	int brick_z = z / BRICK_DIM;
	if (brick_x < 0 || brick_x >= BRICK_MAP_DIM || brick_y < 0 || brick_y >= BRICK_MAP_DIM || brick_z < 0 || brick_z >= BRICK_MAP_DIM) {
		result = Voxel();
		return;
	}
	int grid_index = brick_x + BRICK_MAP_DIM * (brick_y + BRICK_MAP_DIM * brick_z);
	int brickIndex = d_brickMap[grid_index];
	if (brickIndex < 0) {
		result = Voxel();
		return;
	}
	int local_x = x % BRICK_DIM;
	int local_y = y % BRICK_DIM;
	int local_z = z % BRICK_DIM;
	int voxelIndex = local_x + BRICK_DIM * (local_y + BRICK_DIM * local_z);
	result = d_bricks[brickIndex].voxels[voxelIndex];
}


__global__ void updateVoxelKernelClass(BrickMap bm, int x, int y, int z, Voxel newVoxel) {
	bm.deviceUpdateVoxel(x, y, z, newVoxel);
}


//-----------------------------------------------------------------------------
// Kernel to initialize the top-level brick map to –1 (meaning “no brick allocated”).
__global__ void initBrickMapKernel(int* brickMap, int totalEntries) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalEntries) return;

	brickMap[idx] = -1;
}

//-----------------------------------------------------------------------------
// Kernel to initialize the free list. The free list holds indices into the brick pool.
__global__ void initFreeListKernel(int* freeList, int* freeListTop, int maxBricks) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= maxBricks) return;

	if (idx == 0) *freeListTop = maxBricks;

	freeList[idx] = idx;
}

//-----------------------------------------------------------------------------
// Kernel to build the brick map from an array of voxel updates.
__global__ void buildBrickMapKernel(const VoxelUpdate* updates, int numUpdates, int* brickMap, Brick* bricks, int* freeList,
                                           int* freeListTop) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numUpdates) return;

	VoxelUpdate update = updates[idx];

	// Compute the brick coordinates and verify bounds.
	int brick_x = update.x / BRICK_DIM;
	int brick_y = update.y / BRICK_DIM;
	int brick_z = update.z / BRICK_DIM;
	if (brick_x < 0 || brick_x >= BRICK_MAP_DIM || brick_y < 0 || brick_y >= BRICK_MAP_DIM || brick_z < 0 || brick_z >= BRICK_MAP_DIM) {
		return;  // Out-of-bounds update.
	}
	int grid_index = brick_x + BRICK_MAP_DIM * (brick_y + BRICK_MAP_DIM * brick_z);

	// Attempt to reserve a brick (–1 means unallocated; use –2 as allocation in progress).
	int brickIndex = atomicCAS(&brickMap[grid_index], -1, -2);
	if (brickIndex == -1) {
		// We won the race to allocate: pop a brick index from the free list.
		int freeCount = atomicSub(freeListTop, 1);
		if (freeCount <= 0) {
			// Out-of-memory.
			atomicAdd(freeListTop, 1);
			return;
		}
		int newBrickIndex = freeList[freeCount - 1];

		// Initialize the brick: set every voxel to empty and occupancy to 0.
		Brick& brick = bricks[newBrickIndex];
		for (int i = 0; i < VOXELS_PER_BRICK; i++) {
			brick.voxels[i] = Voxel();
		}		brick.occupancy = 0;

		// Write the new brick index into the grid.
		atomicExch(&brickMap[grid_index], newBrickIndex);
		brickIndex = newBrickIndex;
	} else {
		// If allocation is in progress, spin until a valid brick index is available.
		while (brickIndex == -2) brickIndex = brickMap[grid_index];
	}

	// Compute the local voxel index inside the brick.
	Brick* brick = &bricks[brickIndex];
	int local_x = update.x % BRICK_DIM;
	int local_y = update.y % BRICK_DIM;
	int local_z = update.z % BRICK_DIM;
	int voxelIndex = local_x + BRICK_DIM * (local_y + BRICK_DIM * local_z);

	// Update the voxel and its occupancy.
	Voxel oldVoxel = brick->voxels[voxelIndex];
	brick->voxels[voxelIndex] = update.voxel;
	bool wasEmpty = Voxel::isEmpty(oldVoxel);
	bool isNowEmpty = Voxel::isEmpty(update.voxel);
	if (wasEmpty && !isNowEmpty)
		atomicAdd(&brick->occupancy, 1);
	else if (!wasEmpty && isNowEmpty)
		atomicSub(&brick->occupancy, 1);
}

//-----------------------------------------------------------------------------
// A cleanup kernel that scans the grid and deallocates bricks whose occupancy is zero.
__global__ void cleanupEmptyBricksKernel(int* brickMap, Brick* bricks, int* freeList, int* freeListTop, const uint32_t maxBricks) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= TOTAL_BRICKS_IN_MAP) return;

	int brickIndex = brickMap[idx];
	if (brickIndex >= 0)
	{
		Brick* brick = &bricks[brickIndex];
		if (brick->occupancy == 0)
		{
			// Mark this grid cell as unallocated.
			brickMap[idx] = -1;

			// Return the brick to the free list. We do not guard freeListTop here,
			// but if the scene can truly free more bricks than maxBricks, add a check:
			uint32_t pos = atomicAdd(freeListTop, 1);
			if (pos < maxBricks)  // optional guard
			{
				freeList[pos] = brickIndex;
			}
			// else: out-of-range, handle error or clamp
		}
	}
}
