#include <device_atomic_functions.h>

#include <cuda/std/cmath>

#include "BrickMap.cuh"
#include "Sampler.cuh"

__global__ void init_bricks(Brick* bricks, Voxel* voxel_base, const uint32_t brick_volume, const uint32_t maxBricks) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < maxBricks) {
		bricks[idx].data = &voxel_base[idx * brick_volume];
		bricks[idx].isLoaded = false;
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
	if (m_freeIndices.empty()) return BrickConfig::INACTIVE;
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


BrickMap::BrickMap(const Dim dim) : m_dimensions(dim), m_pool(new BrickPool(dim[0] * dim[1] * dim[2])), d_grid(nullptr) {
	// Initialize allocation request buffer
	CHECK_CUDA(cudaMalloc(&d_allocationRequests, sizeof(AllocationRequest) * BrickConfig::MAX_ALLOCATION_REQUESTS));
	CHECK_CUDA(cudaMalloc(&d_requestCounter, sizeof(RequestCounter)));

	//) Initialize counter to 0
	constexpr RequestCounter initialCounter{};
	CHECK_CUDA(cudaMemcpy(d_requestCounter, &initialCounter, sizeof(RequestCounter), cudaMemcpyHostToDevice));

	const uint32_t gridSize = dim[0] * dim[1] * dim[2];
	// Allocate device memory for the grid.
	CHECK_CUDA(cudaMalloc(&d_grid, sizeof(uint32_t) * gridSize));

	// Initialize all grid cells to indicate "empty" (UINT32_MAX).
	CHECK_CUDA(cudaMemset(d_grid, BrickConfig::INACTIVE, sizeof(uint32_t) * gridSize));

	// Allocate memory for d_emptyBricks to use in deallocateInactive()
	CHECK_CUDA(cudaMalloc(&d_emptyBricks, sizeof(EmptyBrickInfo) * gridSize));
}

BrickMap::~BrickMap() {
	if (d_grid) cudaFree(d_grid);
	if (d_allocationRequests) cudaFree(d_allocationRequests);
	if (d_requestCounter) cudaFree(d_requestCounter);
	if (d_emptyBricks) cudaFree(d_emptyBricks);
	delete m_pool;
}


bool BrickMap::allocateBrickAt(const BrickCoord& coord) const {
	if (!BrickMath::isValidBrickCoord(coord, m_dimensions)) return false;

	const uint32_t gridIndex = BrickMath::getBrickIndex(coord, m_dimensions);
	const uint32_t brickIndex = m_pool->allocateBrick();
	if (brickIndex == BrickConfig::INACTIVE) return false;

	// Instead of copying the whole brick structure, simply update the isLoaded flag.
	bool loaded = true;
	CHECK_CUDA(cudaMemcpy(&m_pool->getDeviceBricks()[brickIndex].isLoaded, &loaded, sizeof(bool), cudaMemcpyHostToDevice));

	// Update the grid cell with the allocated brick index.
	CHECK_CUDA(cudaMemcpy(&d_grid[gridIndex], &brickIndex, sizeof(uint32_t), cudaMemcpyHostToDevice));
	return true;
}


__global__ void checkEmptyBricksKernel(const uint32_t* grid, const Brick* bricks, const Dim dim, EmptyBrickInfo* emptyBricks) {
	const uint32_t gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gridIdx >= dim[0] * dim[1] * dim[2]) return;

	const uint32_t brickIdx = grid[gridIdx];
	if (brickIdx == BrickConfig::INACTIVE || !bricks[brickIdx].isLoaded) {
		return;
	}

	const Voxel* brickData = bricks[brickIdx].data;

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

__global__ void kernelBuildBrickMap(const VoxelCoordGlobal* coords, const float* values, const size_t count, const Dim grid_dim,
                                    const Brick* bricks, uint32_t* grid, AllocationRequest* allocationRequests,
                                    RequestCounter* requestCounter) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;

	// Convert OpenVDB coord to brick coordinates
	const VoxelCoordGlobal coord = coords[idx];
	const BrickCoord brick_coord = BrickMath::getBrickCoord(coord);

	// Convert to local voxel coordinates
	const auto local_coord = BrickMath::globalToLocalCoord(coord);

	const uint32_t voxel_idx = BrickMath::localToVoxelIdx(local_coord);
	const uint32_t grid_idx = BrickMath::getBrickIndex(brick_coord, grid_dim);
	const uint32_t brick_index = grid[grid_idx];


	if (brick_index == BrickConfig::INACTIVE) {
		// Use atomicCAS to avoid duplicates
		uint32_t oldVal = atomicCAS(&grid[grid_idx], BrickConfig::INACTIVE, BrickConfig::ALLOC_PENDING);
		if (oldVal == BrickConfig::INACTIVE) {
			// We successfully claimed it, so request a host-side allocation
			BrickMap::requestAllocation(brick_coord, allocationRequests, requestCounter);
		}
		return;  // skip writing voxel data
	}

	if (brick_index == BrickConfig::ALLOC_PENDING) {
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
	VoxelCoordGlobal* d_coords = nullptr;

	CHECK_CUDA(cudaMalloc(&d_values, sizeof(float) * data.size()));
	CHECK_CUDA(cudaMalloc(&d_coords, sizeof(VoxelCoordGlobal) * data.size()));

	CHECK_CUDA(cudaMemcpy(d_values, values.data(), sizeof(float) * data.size(), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_coords, coords.data(), sizeof(VoxelCoordGlobal) * data.size(), cudaMemcpyHostToDevice));

	constexpr uint32_t brick_size = BrickConfig::BRICK_SIZE;

	constexpr uint32_t blockSize = 256;
	const uint32_t gridSize = (data.size() + blockSize - 1) / blockSize;

	// First pass: Request allocations for needed bricks
	kernelBuildBrickMap<<<gridSize, blockSize>>>(d_coords, d_values, data.size(), m_dimensions, m_pool->getDeviceBricks(), d_grid,
	                                             d_allocationRequests, d_requestCounter);

	cudaDeviceSynchronize();  // Ensure kernel has finished

	// Process allocation requests on the host
	processAllocationRequests();


	// Second pass: Write the actual voxel data now that bricks are allocated
	kernelBuildBrickMap<<<gridSize, blockSize>>>(d_coords, d_values, data.size(), m_dimensions, m_pool->getDeviceBricks(), d_grid,
	                                             d_allocationRequests, d_requestCounter);


	// Cleanup
	cudaFree(d_values);
	cudaFree(d_coords);
}


void BrickMap::deallocateBrickAt(const BrickCoord& coord) const {
	if (!BrickMath::isValidBrickCoord(coord, m_dimensions)) return;

	const uint32_t gridIndex = BrickMath::getBrickIndex(coord, m_dimensions);
	uint32_t brickIndex;
	CHECK_CUDA(cudaMemcpy(&brickIndex, &d_grid[gridIndex], sizeof(uint32_t), cudaMemcpyDeviceToHost));

	if (brickIndex != BrickConfig::INACTIVE) {
		m_pool->deallocateBrick(brickIndex);
		brickIndex = BrickConfig::INACTIVE;
		CHECK_CUDA(cudaMemcpy(&d_grid[gridIndex], &brickIndex, sizeof(uint32_t), cudaMemcpyHostToDevice));
	}
}

void BrickMap::deallocateInactive() const {
	const size_t gridSize = m_dimensions[0] * m_dimensions[1] * m_dimensions[2];

	// Launch kernel to check for empty bricks
	dim3 blockSize(256);
	dim3 gridDim((gridSize + blockSize.x - 1) / blockSize.x);
	checkEmptyBricksKernel<<<gridDim, blockSize>>>(d_grid, m_pool->getDeviceBricks(), m_dimensions, d_emptyBricks);

	// Copy results back to host (consider that for a huge grid this copy can be heavy).
	std::vector<EmptyBrickInfo> emptyBricks(gridSize);
	CHECK_CUDA(cudaMemcpy(emptyBricks.data(), d_emptyBricks, sizeof(EmptyBrickInfo) * gridSize, cudaMemcpyDeviceToHost));

	// Process empty bricks
	for (const auto& [gridIndex, isEmpty] : emptyBricks) {
		if (isEmpty) {
			// Get the brick index before deallocation
			uint32_t brickIdx;
			CHECK_CUDA(cudaMemcpy(&brickIdx, &d_grid[gridIndex], sizeof(uint32_t), cudaMemcpyDeviceToHost));

			if (brickIdx != BrickConfig::INACTIVE) {
				// Return the brick to the pool
				m_pool->deallocateBrick(brickIdx);

				// Mark the grid cell as inactive.
				uint32_t inactive = BrickConfig::INACTIVE;
				CHECK_CUDA(cudaMemcpy(&d_grid[gridIndex], &inactive, sizeof(uint32_t), cudaMemcpyHostToDevice));

				printf("Deallocated brick at grid index %u\n", gridIndex);
			}
		}
	}
}


__device__ void BrickMap::requestAllocation(const BrickCoord& coord, AllocationRequest* req, RequestCounter* counter) {
	const uint32_t requestIdx = atomicAdd(&counter->count, 1);
	if (requestIdx < BrickConfig::MAX_ALLOCATION_REQUESTS) {
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
					printf("Failed to allocate brick at (%u, %u, %u)\n", coord[0], coord[1], coord[2]);
				}
			}
		}

		// Reset counter
		constexpr RequestCounter resetCounter{};
		CHECK_CUDA(cudaMemcpy(d_requestCounter, &resetCounter, sizeof(RequestCounter), cudaMemcpyHostToDevice));
	}
}


__global__ void gatherActiveBricksKernel(const uint32_t* d_grid, const Brick* d_bricks, const Dim dim, BrickCoord* outCoords,
                                         uint32_t* outCount) {
	// 1D index
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t total = dim[0] * dim[1] * dim[2];
	if (idx >= total) return;

	// Read the brick index
	uint32_t brickIdx = d_grid[idx];
	if (brickIdx == BrickConfig::INACTIVE) {
		return;  // skip
	}

	// Check if brick is loaded
	const Brick& b = d_bricks[brickIdx];
	if (!b.isLoaded) {
		return;  // skip
	}

	const BrickCoord coord = BrickMath::getBrickGridIdxToCoord(idx, dim);

	// We found an active brick => claim a slot
	uint32_t outPos = atomicAdd(outCount, 1);
	outCoords[outPos] = coord;
}


std::vector<BrickCoord> BrickMap::getActiveBricks() const {
	uint32_t maxPossibleBricks = m_dimensions[0] * m_dimensions[1] * m_dimensions[2];

	// Allocate array of BrickCoord3D
	BrickCoord* d_activeCoords = nullptr;
	CHECK_CUDA(cudaMalloc(&d_activeCoords, maxPossibleBricks * sizeof(BrickCoord)));

	// Allocate a counter on device
	uint32_t* d_count = nullptr;
	CHECK_CUDA(cudaMalloc(&d_count, sizeof(uint32_t)));
	CHECK_CUDA(cudaMemset(d_count, 0, sizeof(uint32_t)));


	uint32_t blockSize = 256;
	uint32_t gridSize = (maxPossibleBricks + blockSize - 1) / blockSize;

	gatherActiveBricksKernel<<<gridSize, blockSize>>>(d_grid, m_pool->getDeviceBricks(), m_dimensions, d_activeCoords, d_count);

	uint32_t activeCount = 0;
	CHECK_CUDA(cudaMemcpy(&activeCount, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	std::vector<BrickCoord> hostActiveCoords(activeCount);

	CHECK_CUDA(cudaMemcpy(hostActiveCoords.data(), d_activeCoords, activeCount * sizeof(BrickCoord), cudaMemcpyDeviceToHost));

	cudaFree(d_activeCoords);
	cudaFree(d_count);

	return hostActiveCoords;
}


Voxel* BrickMap::getBrickAtHost(const BrickCoord& coord) const {
	if (!BrickMath::isValidBrickCoord(coord, m_dimensions)) {
		printf("Coordinates out of bounds\n");
		return nullptr;
	}

	// Get the brick index from the grid
	uint32_t brickIdx;
	CHECK_CUDA(cudaMemcpy(&brickIdx, d_grid + BrickMath::getBrickIndex(coord, m_dimensions), sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (brickIdx == BrickConfig::INACTIVE) {
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

	auto voxelData = new Voxel[Brick::VOLUME];
	CHECK_CUDA(cudaMemcpy(voxelData, brick.data, sizeof(Voxel) * Brick::VOLUME, cudaMemcpyDeviceToHost));

	return voxelData;
}


Voxel* BrickMap::getBrickAtDevice(const BrickCoord& coord) const {
	if (!BrickMath::isValidBrickCoord(coord, m_dimensions)) {
		printf("Coordinates out of bounds\n");
		return nullptr;
	}

	// Get the brick index from the grid
	const uint32_t* brickIdx = d_grid + BrickMath::getBrickIndex(coord, m_dimensions);
	if (*brickIdx == BrickConfig::INACTIVE) {
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

__global__ void accessBrickKernel(const uint32_t* grid, const Brick* bricks, const uint32_t allocatedBricks) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= allocatedBricks) return;

	const uint32_t brickIdx = grid[idx];
	if (brickIdx == BrickConfig::INACTIVE) return;  // Empty cell

	const Brick brick = bricks[brickIdx];
	if (!brick.isLoaded) return;  // Brick not loaded

	Voxel* brickData = brick.data;

	for (uint32_t voxelIdx = 0; voxelIdx < Brick::VOLUME; voxelIdx++) {
		Voxel& voxel = brickData[voxelIdx];
		voxel.density = 1;
		voxel.velocity = nanovdb::Vec3f(1, 0, 0);
	}
}

__global__ void initVel(const uint32_t* grid, const Brick* bricks, const uint32_t allocatedBricks) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= allocatedBricks) return;

	const uint32_t brickIdx = grid[idx];
	if (brickIdx == BrickConfig::INACTIVE) return;  // Empty cell

	const Brick brick = bricks[brickIdx];
	if (!brick.isLoaded) return;  // Brick not loaded

	Voxel* brickData = brick.data;

	for (uint32_t voxelIdx = 0; voxelIdx < Brick::VOLUME; voxelIdx++) {
		Voxel& voxel = brickData[voxelIdx];
		voxel.velocity = nanovdb::Vec3f(1, 0, 0);
	}
}


__global__ void advectAllocate(uint32_t* grid, const Brick* bricks, const float dt, const Dim dim, AllocationRequest* request,
                               RequestCounter* counter) {
	const uint32_t brickGridIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (brickGridIdx >= dim[0] * dim[1] * dim[2]) return;

	const uint32_t brickIdx = grid[brickGridIdx];
	if (brickIdx == BrickConfig::INACTIVE) return;

	const Brick& brick = bricks[brickIdx];
	if (!brick.isLoaded) return;

	const BrickCoord brickCoord = BrickMath::getBrickGridIdxToCoord(brickGridIdx, dim);
	const Voxel* brickData = brick.data;


	for (uint32_t voxelIdx = 0; voxelIdx < Brick::VOLUME; ++voxelIdx) {
		const Voxel& currentVoxel = brickData[voxelIdx];
		// Skip voxels with zero density
		if (currentVoxel.density == 0.0f) continue;

		const VoxelCoordLocal localCoord = BrickMath::voxelIdxToLocal(voxelIdx);
		const VoxelCoordGlobal globalCoord = BrickMath::localToGlobalCoord(brickCoord, localCoord);

		const nanovdb::Vec3f velocity = currentVoxel.velocity;

		const VoxelCoordGlobal destGlobal(globalCoord[0] + velocity[0] * dt, globalCoord[1] + velocity[1] * dt,
		                                  globalCoord[2] + velocity[2] * dt);

		const BrickCoord destBrick = BrickMath::getBrickCoord(destGlobal);
		const uint32_t destGridIdx = BrickMath::getBrickIndex(destBrick, dim);

		if (destGridIdx >= dim[0] * dim[1] * dim[2]) continue;

		const uint32_t destBrickIdx = grid[destGridIdx];

		if (destBrickIdx == BrickConfig::INACTIVE) {
			const uint32_t oldVal = atomicCAS(&grid[destGridIdx], BrickConfig::INACTIVE, BrickConfig::ALLOC_PENDING);
			if (oldVal == BrickConfig::INACTIVE) {
				BrickMap::requestAllocation(destBrick, request, counter);
			}
		}
	}
}


__global__ void advectKernel(const uint32_t* grid, const Brick* bricks, const uint32_t d_allocatedBricks, const float dt, const Dim dim) {
	const uint32_t brickGridIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (brickGridIdx >= d_allocatedBricks) return;

	const uint32_t brickIdx = grid[brickGridIdx];
	if (brickIdx == BrickConfig::INACTIVE) return;

	const Brick& brick = bricks[brickIdx];
	if (!brick.isLoaded) return;

	const BrickCoord brickCoord = BrickMath::getBrickGridIdxToCoord(brickGridIdx, dim);

	Voxel* brickData = brick.data;

	for (uint32_t voxelIdx = 0; voxelIdx < Brick::VOLUME; ++voxelIdx) {
		const VoxelCoordLocal localCoord = BrickMath::voxelIdxToLocal(voxelIdx);
		const VoxelCoordGlobal globalCoord = BrickMath::localToGlobalCoord(brickCoord, localCoord);

		const Voxel& currentVoxel = brickData[voxelIdx];
		const nanovdb::Vec3f velocity = currentVoxel.velocity;


		// Convert to float coordinates for more precise advection
		const float sourceX = float(globalCoord[0]) - velocity[0] * dt;
		const float sourceY = float(globalCoord[1]) - velocity[1] * dt;
		const float sourceZ = float(globalCoord[2]) - velocity[2] * dt;

		const nanovdb::Vec3f sourceCoord(sourceX, sourceY, sourceZ);

		const Voxel& sampled = trilinearSample(grid, bricks, dim, sourceCoord);

		brickData[voxelIdx].density = sampled.density;
		brickData[voxelIdx].velocity = sampled.velocity;
	}
}


extern "C" void advect(const BrickMap& brickMap, const float dt) {
	uint32_t* d_grid = brickMap.getDeviceGrid();
	const Brick* d_bricks = brickMap.getPool()->getDeviceBricks();
	const Dim dim = brickMap.getDimensions();
	AllocationRequest* d_requests = brickMap.getDeviceAllocationRequests();
	RequestCounter* d_counter = brickMap.getDeviceRequestCounter();
	uint32_t d_allocatedBricks = brickMap.getPool()->getNumAllocatedBricks();


	uint32_t blockSize = 256;
	uint32_t gridSize = (d_allocatedBricks + blockSize - 1) / blockSize;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	advectAllocate<<<gridSize, blockSize, 0, stream>>>(d_grid, d_bricks, dt, dim, d_requests, d_counter);

	CHECK_CUDA(cudaStreamSynchronize(stream));

	brickMap.processAllocationRequests();

	initVel<<<gridSize, blockSize, 0, stream>>>(d_grid, d_bricks, d_allocatedBricks);

	advectKernel<<<gridSize, blockSize, 0, stream>>>(d_grid, d_bricks, d_allocatedBricks, dt, dim);
}


extern "C" void accessBrick(const BrickMap& brickMap) {
	const uint32_t* d_grid = brickMap.getDeviceGrid();
	const Brick* d_bricks = brickMap.getPool()->getDeviceBricks();
	const Dim dim = brickMap.getDimensions();
	const uint32_t d_allocatedBricks = brickMap.getPool()->getNumAllocatedBricks();
	uint32_t blockSize = 256;
	uint32_t gridSize = (dim[0] * dim[1] * dim[2] + blockSize - 1) / blockSize;
	accessBrickKernel<<<gridSize, blockSize>>>(d_grid, d_bricks, d_allocatedBricks);
}


extern "C" void InitVel(const BrickMap& brickMap) {
	const uint32_t* d_grid = brickMap.getDeviceGrid();
	const Brick* d_bricks = brickMap.getPool()->getDeviceBricks();
	const Dim dim = brickMap.getDimensions();
	const uint32_t d_allocatedBricks = brickMap.getPool()->getNumAllocatedBricks();
	uint32_t blockSize = 256;
	uint32_t gridSize = (dim[0] * dim[1] * dim[2] + blockSize - 1) / blockSize;
	initVel<<<gridSize, blockSize>>>(d_grid, d_bricks, d_allocatedBricks);
}


__global__ void debugSampleKernel(const uint32_t* grid, const Brick* bricks, const Dim dim) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		const nanovdb::Vec3f a(31.5, 31.5, 31.5);
		Voxel v1 = trilinearSample(grid, bricks, dim, a);
		printf("Sample @ (%f, %f, %f): density=%.3f\n", 31.5, 31.5, 31.5, v1.density);
		printf("Sample @ (%f, %f, %f): velocity={%.3f, %.3f, %.3f}\n", 31.5, 31.5, 31.5, v1.velocity[0], v1.velocity[1], v1.velocity[2]);

		const nanovdb::Vec3f b(31, 31, 31);
		Voxel v2 = trilinearSample(grid, bricks, dim, b);
		printf("Sample @ (%d, %d, %d): density=%.3f\n", 31, 31, 31, v2.density);
		printf("Sample @ (%d, %d, %d): velocity={%.3f, %.3f, %.3f}\n", 31, 31, 31, v2.velocity[0], v2.velocity[1], v2.velocity[2]);
	}
}