#include "BrickMap.cuh"

BrickPool::BrickPool(const uint32_t maxBricks) : m_maxBricks(maxBricks), m_numAllocatedBricks(0), d_bricks(nullptr), d_voxelData(nullptr) {
	cudaMalloc(&d_bricks, m_maxBricks * sizeof(Brick));
	cudaMalloc(&d_voxelData, m_maxBricks * Brick::VOLUME * sizeof(Voxel));

	std::vector<Brick> hostBricks(maxBricks);
	for (uint32_t i = 0; i < maxBricks; ++i) {
		hostBricks[i].isLoaded = false;
		hostBricks[i].poolIndex = i;
		hostBricks[i].data = d_voxelData + i * Brick::VOLUME;
	}

	cudaMemcpy(d_bricks, hostBricks.data(), sizeof(Brick) * maxBricks, cudaMemcpyHostToDevice);

	m_freeIndices.resize(maxBricks);
	for (uint32_t i = 0; i < maxBricks; ++i) {
		m_freeIndices[i] = maxBricks - i - 1;
	}
}


BrickPool::~BrickPool() {
	if (d_bricks) cudaFree(d_bricks);
	if (d_voxelData) cudaFree(d_voxelData);
}


uint32_t BrickPool::allocateBrick() {
	if (m_freeIndices.empty()) return UINT32_MAX;
	const uint32_t index = m_freeIndices.back();
	m_freeIndices.pop_back();
	m_numAllocatedBricks++;
	return index;
}


void BrickPool::deallocateBrick(const uint32_t index) {
	if (index >= m_maxBricks) return;
	m_freeIndices.push_back(index);
	m_numAllocatedBricks--;
}


BrickMap::BrickMap(const uint32_t dimX, const uint32_t dimY, const uint32_t dimZ)
    : m_dimensions{dimX, dimY, dimZ}, m_pool(std::make_unique<BrickPool>()), d_grid(nullptr) {
	// Initialize allocation request buffer
	cudaMalloc(&d_allocationRequests, sizeof(AllocationRequest) * MAX_ALLOCATION_REQUESTS);
	cudaMalloc(&d_requestCounter, sizeof(RequestCounter));

	// Initialize counter to 0
	RequestCounter initialCounter = {0};
	cudaMemcpy(d_requestCounter, &initialCounter, sizeof(RequestCounter), cudaMemcpyHostToDevice);

	const uint32_t gridSize = dimX * dimY * dimZ;
	// Allocate device memory for the grid.
	cudaMalloc(&d_grid, sizeof(uint32_t) * gridSize);

	// Initialize all grid cells to indicate "empty" (UINT32_MAX).
	cudaMemset(d_grid, UINT32_MAX, sizeof(uint32_t) * gridSize);
}


BrickMap::~BrickMap() {
	if (d_grid) cudaFree(d_grid);
	if (d_allocationRequests) cudaFree(d_allocationRequests);
	if (d_requestCounter) cudaFree(d_requestCounter);
}


bool BrickMap::allocateBrickAt(const uint32_t x, const uint32_t y, const uint32_t z) const {
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) return false;

	const uint32_t gridIndex = getGridIndex(x, y, z);

	// (Optional) Check if already allocated here.
	const uint32_t index = m_pool->allocateBrick();
	if (index == UINT32_MAX) return false;

	// Update the brick structure on the device to mark it as loaded.
	Brick brick{};
	brick.isLoaded = true;
	brick.poolIndex = index;
	brick.data = m_pool->getDeviceVoxelData() + index * Brick::VOLUME;

	cudaMemcpy(m_pool->getDeviceBricks() + index, &brick, sizeof(Brick), cudaMemcpyHostToDevice);

	// Write the allocated brick index into the grid cell.
	cudaMemcpy(d_grid + gridIndex, &index, sizeof(uint32_t), cudaMemcpyHostToDevice);

	return true;
}


void BrickMap::deallocateBrickAt(const uint32_t x, const uint32_t y, const uint32_t z) const {
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) return;

	const uint32_t gridIndex = getGridIndex(x, y, z);
	uint32_t brickIndex;
	cudaMemcpy(&brickIndex, d_grid + gridIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	if (brickIndex != UINT32_MAX) {
		m_pool->deallocateBrick(brickIndex);
		brickIndex = UINT32_MAX;
		cudaMemcpy(d_grid + gridIndex, &brickIndex, sizeof(uint32_t), cudaMemcpyHostToDevice);
	}
}


__device__ void BrickMap::requestAllocation(uint32_t x, uint32_t y, uint32_t z, AllocationRequest* req, RequestCounter* counter) {
	uint32_t requestIdx = atomicAdd(&counter->count, 1);
	if (requestIdx < MAX_ALLOCATION_REQUESTS) {
		req[requestIdx] = {x, y, z, true};
	}
}


__host__ void BrickMap::processAllocationRequests() {
	// Read request counter
	RequestCounter counter{};
	cudaMemcpy(&counter, d_requestCounter, sizeof(RequestCounter), cudaMemcpyDeviceToHost);

	if (counter.count > 0) {
		// Read allocation requests
		std::vector<AllocationRequest> requests(counter.count);
		cudaMemcpy(requests.data(), d_allocationRequests, sizeof(AllocationRequest) * counter.count, cudaMemcpyDeviceToHost);

		// Process each request
		for (const auto& [x, y, z, valid] : requests) {
			if (valid) {
				if (!allocateBrickAt(x, y, z)) {
					printf("Failed to allocate brick at (%u, %u, %u)\n", x, y, z);
				}
			}
		}

		// Reset counter
		constexpr RequestCounter resetCounter{};
		cudaMemcpy(d_requestCounter, &resetCounter, sizeof(RequestCounter), cudaMemcpyHostToDevice);
	}
}

Voxel* BrickMap::getVoxelAt(const uint32_t x, const uint32_t y, const uint32_t z) const {
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) return nullptr;

	uint32_t brickIdx;
	cudaMemcpy(&brickIdx, d_grid + getGridIndex(x, y, z), sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (brickIdx == UINT32_MAX) {
		printf("Empty cell\n");
		return nullptr;  // Empty cell
	}

	// Copy the brick structure from device memory.
	Brick brick;
	cudaMemcpy(&brick, m_pool->getDeviceBricks() + brickIdx, sizeof(Brick), cudaMemcpyDeviceToHost);
	if (!brick.isLoaded) {
		printf("Brick not loaded\n");
		return nullptr;  // Brick not loaded
	}

	// Compute pointer to the voxel data for this brick.
	Voxel* brickData = m_pool->getDeviceVoxelData() + brickIdx * Brick::VOLUME;

	Voxel* h_voxel = nullptr;
	cudaMallocHost(&h_voxel, sizeof(Voxel));
	cudaMemcpy(h_voxel, brickData, sizeof(Voxel), cudaMemcpyDeviceToHost);

	return h_voxel;
}

__global__ void accessBrickKernel(const uint32_t* grid, const Brick* bricks, Voxel* voxelData, const uint32_t dimX, const uint32_t dimY,
                                  const uint32_t dimZ) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t totalCells = dimX * dimY * dimZ;
	if (idx >= totalCells) return;

	uint32_t brickIdx = grid[idx];
	if (brickIdx == UINT32_MAX) return;  // Empty cell

	// Copy the brick structure from device memory.
	Brick brick = bricks[brickIdx];
	if (!brick.isLoaded) return;  // Brick not loaded

	// Compute pointer to the voxel data for this brick.
	Voxel* brickData = &voxelData[brickIdx * Brick::VOLUME];

	brickData[0].density = 1.0f;
	brickData[0].temperature = 1.0f;
	brickData[0].fuel = 1.0f;
	brickData[0].velocity = nanovdb::Vec3f(1.0f);
}

extern "C" void accessBrick(const BrickMap& brickMap) {
	printf("Kernel Launched\n");
	const uint32_t* d_grid = brickMap.getDeviceGrid();
	const Brick* d_bricks = brickMap.getPool()->getDeviceBricks();
	Voxel* d_voxelData = brickMap.getPool()->getDeviceVoxelData();
	const uint32_t* dim = brickMap.getDimensions();

	dim3 blockSize(256);
	dim3 gridSize((256 * 256 * 256 + blockSize.x - 1) / blockSize.x);

	accessBrickKernel<<<gridSize, blockSize>>>(d_grid, d_bricks, d_voxelData, dim[0], dim[1], dim[2]);
}