#include "BrickMap.cuh"


__global__ void init_bricks(Brick* bricks, Voxel* voxel_base, uint32_t brick_volume) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < brick_volume) {
		bricks[idx].data = voxel_base + idx * brick_volume;
		bricks[idx].isLoaded = false;
		bricks[idx].poolIndex = idx;
	}
}

BrickPool::BrickPool(const uint32_t maxBricks) : m_maxBricks(maxBricks), m_numAllocatedBricks(0), d_bricks(nullptr), d_voxelData(nullptr) {
	cudaMalloc(&d_bricks, m_maxBricks * sizeof(Brick));
	cudaMalloc(&d_voxelData, m_maxBricks * Brick::VOLUME * sizeof(Voxel));

	uint32_t blockSize = 256;
	uint32_t gridSize = (m_maxBricks + blockSize - 1) / blockSize;
	init_bricks<<<gridSize, blockSize>>>(d_bricks, d_voxelData, Brick::VOLUME);

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

	const uint32_t gridIndex = getBrickIndex(x, y, z);

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

	const uint32_t gridIndex = getBrickIndex(x, y, z);
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


Voxel* BrickMap::getBrickAtHost(const uint32_t x, const uint32_t y, const uint32_t z) const {
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) {
		printf("Coordinates out of bounds\n");
		return nullptr;
	}

	// Get the brick index from the grid
	uint32_t brickIdx;
	cudaMemcpy(&brickIdx, d_grid + getBrickIndex(x, y, z), sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (brickIdx == UINT32_MAX) {
		printf("Empty cell\n");
		return nullptr;  // Empty cell
	}

	// Copy the brick structure from device memory
	Brick brick;
	cudaMemcpy(&brick, m_pool->getDeviceBricks() + brickIdx, sizeof(Brick), cudaMemcpyDeviceToHost);
	if (!brick.isLoaded) {
		printf("Brick not loaded\n");
		return nullptr;  // Brick not loaded
	}

	// Allocate host memory for the entire brick's voxel data
	Voxel* h_brickData = nullptr;
	cudaMallocHost(&h_brickData, Brick::VOLUME * sizeof(Voxel));  // Allocate pinned host memory

	cudaMemcpy(h_brickData, m_pool->getDeviceVoxelData() + brickIdx * Brick::VOLUME, Brick::VOLUME * sizeof(Voxel), cudaMemcpyDeviceToHost);

	return h_brickData;
}


Voxel* BrickMap::getBrickAtDevice(const uint32_t x, const uint32_t y, const uint32_t z) const {
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) {
		printf("Coordinates out of bounds\n");
		return nullptr;
	}

	// Get the brick index from the grid
	uint32_t* brickIdx = d_grid + getBrickIndex(x, y, z);
	if (*brickIdx == UINT32_MAX) {
		printf("Empty cell\n");
		return nullptr;  // Empty cell
	}


	// Copy the brick structure from device memory
	const Brick* brick = m_pool->getDeviceBricks() + *brickIdx;
	if (!brick->isLoaded) {
		printf("Brick not loaded\n");
		return nullptr;  // Brick not loaded
	}

	return brick->data;
}

__global__ void advectionKernel(uint32_t* grid, Brick* bricks, Voxel* voxelData, uint32_t dimX, uint32_t dimY, uint32_t dimZ,
                                AllocationRequest* requests, RequestCounter* counter, float dt, float3 velocity) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dimX * dimY * dimZ) return;

	// Calculate 3D position
	uint32_t z = idx / (dimX * dimY);
	uint32_t y = (idx % (dimX * dimY)) / dimX;
	uint32_t x = idx % dimX;

	// Calculate advection source position
	float3 pos = make_float3(x - velocity.x * dt, y - velocity.y * dt, z - velocity.z * dt);

	// Find source grid cell
	int srcX = __float2int_rd(pos.x);
	int srcY = __float2int_rd(pos.y);
	int srcZ = __float2int_rd(pos.z);

	// Check if source position is valid and has data
	if (srcX >= 0 && srcX < dimX && srcY >= 0 && srcY < dimY && srcZ >= 0 && srcZ < dimZ) {
		uint32_t srcIdx = srcX + srcY * dimX + srcZ * dimX * dimY;
		uint32_t srcBrickIdx = grid[srcIdx];

		// If source has data but destination doesn't, request allocation
		if (srcBrickIdx != UINT32_MAX && grid[idx] == UINT32_MAX) {
			BrickMap::requestAllocation(x, y, z, requests, counter);
		}
	}
}


__global__ void accessBrickKernel(const uint32_t* grid, const Brick* bricks, Voxel* voxelData, const uint32_t dimX, const uint32_t dimY,
                                  const uint32_t dimZ) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t totalCells = dimX * dimY * dimZ;

	if (idx >= totalCells) return;

	const uint32_t brickIdx = grid[idx];
	if (brickIdx == UINT32_MAX) return;  // Empty cell

	const Brick brick = bricks[brickIdx];
	if (!brick.isLoaded) return;  // Brick not loaded

	Voxel* brickData = &voxelData[brickIdx * Brick::VOLUME];

	for (uint32_t voxelIdx = 0; voxelIdx < Brick::VOLUME; voxelIdx++) {
		Voxel& voxel = brickData[voxelIdx];
		voxel.density = voxelIdx;
	}
}


extern "C" void accessBrick(const BrickMap& brickMap) {
	printf("Kernel Launched\n");
	const uint32_t* d_grid = brickMap.getDeviceGrid();
	const Brick* d_bricks = brickMap.getPool()->getDeviceBricks();
	Voxel* d_voxelData = brickMap.getPool()->getDeviceVoxelData();
	const uint32_t* dim = brickMap.getDimensions();

	uint32_t blockSize = 256;
	uint32_t gridSize = (dim[0] * dim[1] * dim[2] + blockSize - 1) / blockSize;
	accessBrickKernel<<<gridSize, blockSize>>>(d_grid, d_bricks, d_voxelData, dim[0], dim[1], dim[2]);
}