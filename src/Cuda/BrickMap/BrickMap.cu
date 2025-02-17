#include <device_atomic_functions.h>

#include "BrickMap.cuh"

__global__ void init_bricks(Brick* bricks, Voxel* voxel_base, const uint32_t brick_volume, const uint32_t maxBricks) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < maxBricks) {
		bricks[idx].data = voxel_base + idx * brick_volume;
		bricks[idx].isLoaded = false;
		bricks[idx].poolIndex = idx;
	}
}

BrickPool::BrickPool(const uint32_t maxBricks) : m_maxBricks(maxBricks), m_numAllocatedBricks(0), d_bricks(nullptr), d_voxelData(nullptr) {
	CHECK_CUDA(cudaMalloc(&d_bricks, m_maxBricks * sizeof(Brick)));
	CHECK_CUDA(cudaMalloc(&d_voxelData, m_maxBricks * Brick::VOLUME * sizeof(Voxel)));

	uint32_t blockSize = 256;
	uint32_t gridSize = (m_maxBricks + blockSize - 1) / blockSize;
	init_bricks<<<gridSize, blockSize>>>(d_bricks, d_voxelData, Brick::VOLUME, m_maxBricks);

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
	if (m_freeIndices.empty()) return INACTIVE;
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
	CHECK_CUDA(cudaMalloc(&d_allocationRequests, sizeof(AllocationRequest) * MAX_ALLOCATION_REQUESTS));
	CHECK_CUDA(cudaMalloc(&d_requestCounter, sizeof(RequestCounter)));

	//) Initialize counter to 0
	constexpr RequestCounter initialCounter{};
	CHECK_CUDA(cudaMemcpy(d_requestCounter, &initialCounter, sizeof(RequestCounter), cudaMemcpyHostToDevice));

	const uint32_t gridSize = dimX * dimY * dimZ;
	// Allocate device memory for the grid.
	CHECK_CUDA(cudaMalloc(&d_grid, sizeof(uint32_t) * gridSize));

	// Initialize all grid cells to indicate "empty" (UINT32_MAX).
	cudaMemset(d_grid, INACTIVE, sizeof(uint32_t) * gridSize);

	// Allocate memory for d_emptyBricks to use in deallocateInactive()
	CHECK_CUDA(cudaMalloc(&d_emptyBricks, sizeof(EmptyBrickInfo) * gridSize));
}


BrickMap::~BrickMap() {
	if (d_grid) cudaFree(d_grid);
	if (d_allocationRequests) cudaFree(d_allocationRequests);
	if (d_requestCounter) cudaFree(d_requestCounter);
	if (d_emptyBricks) cudaFree(d_emptyBricks);
}


bool BrickMap::allocateBrickAt(const BrickCoord& coord) const {
	const uint32_t x = coord.x();
	const uint32_t y = coord.y();
	const uint32_t z = coord.z();

	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) return false;

	const uint32_t gridIndex = getBrickIndex(x, y, z);

	// (Optional) Check if already allocated here.
	const uint32_t index = m_pool->allocateBrick();
	if (index == INACTIVE) return false;

	Brick brick;
	CHECK_CUDA(cudaMemcpy(&brick, m_pool->getDeviceBricks() + index, sizeof(Brick), cudaMemcpyDeviceToHost));
	brick.isLoaded = true;

	CHECK_CUDA(cudaMemcpy(m_pool->getDeviceBricks() + index, &brick, sizeof(Brick), cudaMemcpyHostToDevice));

	// Update the grid cell with the allocated brick index.
	CHECK_CUDA(cudaMemcpy(d_grid + gridIndex, &index, sizeof(uint32_t), cudaMemcpyHostToDevice));

	return true;
}


__global__ void checkEmptyBricksKernel(const uint32_t* grid, const Brick* bricks, Voxel* voxelData, const uint32_t dimX,
                                       const uint32_t dimY, const uint32_t dimZ, EmptyBrickInfo* emptyBricks) {
	const uint32_t gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gridIdx >= dimX * dimY * dimZ) return;

	const uint32_t brickIdx = grid[gridIdx];
	if (brickIdx == INACTIVE || !bricks[brickIdx].isLoaded) {
		emptyBricks[gridIdx] = {gridIdx, true};  // Mark as not empty if no brick exists
		return;
	}

	const Voxel* brickData = &voxelData[brickIdx * Brick::VOLUME];

	bool isEmpty = true;

	// Check all voxels in the brick
	for (uint32_t voxelIdx = 0; voxelIdx < Brick::VOLUME; voxelIdx++) {
		const Voxel& voxel = brickData[voxelIdx];

		// Check if any voxel data is non-zero
		if (voxel.density != 0.0f || voxel.temperature != 0.0f || voxel.fuel != 0.0f || voxel.velocity[0] != 0.0f ||
		    voxel.velocity[1] != 0.0f || voxel.velocity[2] != 0.0f) {
			isEmpty = false;
			break;
		}
	}

	emptyBricks[gridIdx] = {gridIdx, isEmpty};
}

__global__ void kernelBuildBrickMap(const VoxelCoord* coords, const float* values, const size_t count, const uint32_t brick_size,
                                    const uint32_t grid_dim, const Brick* bricks, uint32_t* grid, AllocationRequest* allocationRequests,
                                    RequestCounter* requestCounter) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;

	// Convert OpenVDB coord to brick coordinates
	const VoxelCoord coord = coords[idx];
	const auto brick_coord = BrickCoord(coord.x() / brick_size, coord.y() / brick_size, coord.z() / brick_size);

	// Convert to local voxel coordinates
	const auto local_coord = VoxelCoord(coord.x() % brick_size, coord.y() % brick_size, coord.z() % brick_size);

	const uint32_t voxel_idx = local_coord.x() + local_coord.y() * brick_size + local_coord.z() * brick_size * brick_size;
	const uint32_t grid_idx = brick_coord.x() + brick_coord.y() * grid_dim + brick_coord.z() * grid_dim * grid_dim;
	const uint32_t brick_index = grid[grid_idx];


	if (brick_index == UINT32_MAX) {
		// Use atomicCAS to avoid duplicates
		uint32_t oldVal = atomicCAS(&grid[grid_idx], UINT32_MAX, ALLOC_PENDING_MARKER);
		if (oldVal == UINT32_MAX) {
			// We successfully claimed it, so request a host-side allocation
			BrickMap::requestAllocation(brick_coord, allocationRequests, requestCounter);
		}
		return;  // skip writing voxel data
	}

	if (brick_index == ALLOC_PENDING_MARKER) {
		return;
	}


	// Write voxel data if brick exists
	Voxel* voxel = &bricks[brick_index].data[voxel_idx];
	voxel->density = values[idx];
}


void BrickMap::buildFromVDB(const std::vector<std::pair<nanovdb::Coord, float>>& data) const {
	std::vector<float> values;
	std::vector<nanovdb::Coord> coords;
	values.reserve(data.size());
	coords.reserve(data.size());

	for (const auto& [coord, value] : data) {
		coords.push_back(coord);
		values.push_back(value);
	}

	// Allocate device memory
	float* d_values = nullptr;
	VoxelCoord* d_coords = nullptr;

	CHECK_CUDA(cudaMalloc(&d_values, sizeof(float) * data.size()));
	CHECK_CUDA(cudaMalloc(&d_coords, sizeof(VoxelCoord) * data.size()));

	CHECK_CUDA(cudaMemcpy(d_values, values.data(), sizeof(float) * data.size(), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_coords, coords.data(), sizeof(VoxelCoord) * data.size(), cudaMemcpyHostToDevice));

	constexpr uint32_t brick_size = BRICK_SIZE;
	const uint32_t grid_dim = m_dimensions[0];

	constexpr uint32_t blockSize = 256;
	const uint32_t gridSize = (data.size() + blockSize - 1) / blockSize;

	// First pass: Request allocations for needed bricks
	kernelBuildBrickMap<<<gridSize, blockSize>>>(d_coords, d_values, data.size(), brick_size, grid_dim, m_pool->getDeviceBricks(), d_grid,
	                                             d_allocationRequests, d_requestCounter);

	cudaDeviceSynchronize();  // Ensure kernel has finished

	// Process allocation requests on the host
	processAllocationRequests();


	// Second pass: Write the actual voxel data now that bricks are allocated
	kernelBuildBrickMap<<<gridSize, blockSize>>>(d_coords, d_values, data.size(), brick_size, grid_dim, m_pool->getDeviceBricks(), d_grid,
	                                             d_allocationRequests, d_requestCounter);


	// Cleanup
	cudaFree(d_values);
	cudaFree(d_coords);
}


void BrickMap::deallocateBrickAt(const BrickCoord& coord) const {
	const uint32_t x = coord.x();
	const uint32_t y = coord.y();
	const uint32_t z = coord.z();

	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) return;

	const uint32_t gridIndex = getBrickIndex(x, y, z);
	uint32_t brickIndex;
	CHECK_CUDA(cudaMemcpy(&brickIndex, d_grid + gridIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	if (brickIndex != INACTIVE) {
		m_pool->deallocateBrick(brickIndex);
		brickIndex = INACTIVE;
		CHECK_CUDA(cudaMemcpy(d_grid + gridIndex, &brickIndex, sizeof(uint32_t), cudaMemcpyHostToDevice));
	}
}

void BrickMap::deallocateInactive() const {
	const size_t gridSize = m_dimensions[0] * m_dimensions[1] * m_dimensions[2];

	// Launch kernel to check for empty bricks
	dim3 blockSize(256);
	dim3 gridDim((gridSize + blockSize.x - 1) / blockSize.x);

	checkEmptyBricksKernel<<<gridDim, blockSize>>>(d_grid, m_pool->getDeviceBricks(), m_pool->getDeviceVoxelData(), m_dimensions[0],
	                                               m_dimensions[1], m_dimensions[2], d_emptyBricks);

	// Copy results back to host
	std::vector<EmptyBrickInfo> emptyBricks(gridSize);
	CHECK_CUDA(cudaMemcpy(emptyBricks.data(), d_emptyBricks, sizeof(EmptyBrickInfo) * gridSize, cudaMemcpyDeviceToHost));

	// Process empty bricks
	for (const auto& [gridIndex, isEmpty] : emptyBricks) {
		if (isEmpty) {
			// Get the brick index before deallocation
			uint32_t brickIdx;
			CHECK_CUDA(cudaMemcpy(&brickIdx, &d_grid[gridIndex], sizeof(uint32_t), cudaMemcpyDeviceToHost));

			if (brickIdx != INACTIVE) {
				// Return the brick to the pool
				m_pool->deallocateBrick(brickIdx);

				// Mark the grid cell as empty
				CHECK_CUDA(cudaMemcpy(&d_grid[gridIndex], &INACTIVE, sizeof(uint32_t), cudaMemcpyHostToDevice));

				printf("Deallocated brick at grid index %u\n", gridIndex);
			}
		}
	}
}


__device__ void BrickMap::requestAllocation(const BrickCoord& coord, AllocationRequest* req, RequestCounter* counter) {
	const uint32_t requestIdx = atomicAdd(&counter->count, 1);
	if (requestIdx < MAX_ALLOCATION_REQUESTS) {
		req[requestIdx] = {coord, 1};
	}
}


__host__ void BrickMap::processAllocationRequests() const {
	// Read request counter
	RequestCounter counter{};
	CHECK_CUDA(cudaMemcpy(&counter, d_requestCounter, sizeof(RequestCounter), cudaMemcpyDeviceToHost));

	if (counter.count > 0) {
		// Read allocation requests
		std::vector<AllocationRequest> requests(counter.count);
		CHECK_CUDA(cudaMemcpy(requests.data(), d_allocationRequests, sizeof(AllocationRequest) * counter.count, cudaMemcpyDeviceToHost));

		// Process each request
		for (const auto& [coord, valid] : requests) {
			if (valid) {
				if (!allocateBrickAt(coord)) {
					printf("Failed to allocate brick at (%u, %u, %u)\n", coord.x(), coord.y(), coord.z());
				}
			}
		}

		// Reset counter
		constexpr RequestCounter resetCounter{};
		CHECK_CUDA(cudaMemcpy(d_requestCounter, &resetCounter, sizeof(RequestCounter), cudaMemcpyHostToDevice));
	}
}

std::vector<nanovdb::Coord> BrickMap::getActiveBricks() const {
	std::vector<nanovdb::Coord> bricks;
	for (uint32_t z = 0; z < m_dimensions[2]; ++z) {
		for (uint32_t y = 0; y < m_dimensions[1]; ++y) {
			for (uint32_t x = 0; x < m_dimensions[0]; ++x) {
				uint32_t brickIdx;
				CHECK_CUDA(cudaMemcpy(&brickIdx, d_grid + getBrickIndex(x, y, z), sizeof(uint32_t), cudaMemcpyDeviceToHost));
				if (brickIdx == INACTIVE) continue;

				Brick brick;
				CHECK_CUDA(cudaMemcpy(&brick, m_pool->getDeviceBricks() + brickIdx, sizeof(Brick), cudaMemcpyDeviceToHost));

				if (brick.isLoaded) {
					bricks.emplace_back(x, y, z);
				}
			}
		}
	}

	return bricks;
}

Voxel* BrickMap::getBrickAtHost(const uint32_t x, const uint32_t y, const uint32_t z) const {
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) {
		printf("Coordinates out of bounds\n");
		return nullptr;
	}

	// Get the brick index from the grid
	uint32_t brickIdx;
	CHECK_CUDA(cudaMemcpy(&brickIdx, d_grid + getBrickIndex(x, y, z), sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (brickIdx == INACTIVE) {
		printf("Empty cell\n");
		return nullptr;  // Empty cell
	}

	// Copy the brick structure from device memory
	Brick brick;
	CHECK_CUDA(cudaMemcpy(&brick, m_pool->getDeviceBricks() + brickIdx, sizeof(Brick), cudaMemcpyDeviceToHost));
	if (!brick.isLoaded) {
		printf("Brick not loaded\n");
		return nullptr;  // Brick not loaded
	}

	// Allocate host memory for the entire brick's voxel data
	Voxel* h_brickData = nullptr;
	CHECK_CUDA(cudaMallocHost(&h_brickData, Brick::VOLUME * sizeof(Voxel)));  // Allocate pinned host memory

	CHECK_CUDA(cudaMemcpy(h_brickData, m_pool->getDeviceVoxelData() + brickIdx * Brick::VOLUME, Brick::VOLUME * sizeof(Voxel),
	                      cudaMemcpyDeviceToHost));

	return h_brickData;
}

__host__ std::pair<nanovdb::Coord, nanovdb::Coord> BrickMap::getBrickDimensions(uint32_t x, uint32_t y, uint32_t z) const {
	// Check if the brick coordinates are within the brick grid
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) {
		printf("Brick coordinates out of bounds\n");
		return {};
	}

	// Compute the normalized spatial dimensions of the brick
	const auto min = nanovdb::Coord(x, y, z);
	const nanovdb::Coord max = min + nanovdb::Coord(1);

	return {min, max};
}


Voxel* BrickMap::getBrickAtDevice(const uint32_t x, const uint32_t y, const uint32_t z) const {
	if (x >= m_dimensions[0] || y >= m_dimensions[1] || z >= m_dimensions[2]) {
		printf("Coordinates out of bounds\n");
		return nullptr;
	}

	// Get the brick index from the grid
	const uint32_t* brickIdx = d_grid + getBrickIndex(x, y, z);
	if (*brickIdx == INACTIVE) {
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

__global__ void accessBrickKernel(const uint32_t* grid, const Brick* bricks, Voxel* voxelData, const uint32_t dimX, const uint32_t dimY,
                                  const uint32_t dimZ) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t totalCells = dimX * dimY * dimZ;

	if (idx >= totalCells) return;

	const uint32_t brickIdx = grid[idx];
	if (brickIdx == INACTIVE) return;  // Empty cell

	const Brick brick = bricks[brickIdx];
	if (!brick.isLoaded) return;  // Brick not loaded

	Voxel* brickData = &voxelData[brickIdx * Brick::VOLUME];

	for (uint32_t voxelIdx = 0; voxelIdx < Brick::VOLUME; voxelIdx++) {
		Voxel& voxel = brickData[voxelIdx];
		voxel.density = 1.0f * voxelIdx;
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