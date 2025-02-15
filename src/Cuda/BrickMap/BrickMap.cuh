#pragma once

#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <nanovdb/math/Math.h>
#include <nanovdb/util/cuda/Util.h>

#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                                                                     \
	{                                                                                                                        \
		cudaError_t err = call;                                                                                              \
		if (err != cudaSuccess) {                                                                                            \
			std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
			exit(EXIT_FAILURE);                                                                                              \
		}                                                                                                                    \
	}


constexpr uint32_t BRICK_SIZE = 32;
constexpr uint32_t DEFAULT_GRID_DIM = 256;
constexpr size_t MAX_BRICKS = 128 * 128;
constexpr size_t MAX_ALLOCATION_REQUESTS = MAX_BRICKS;
constexpr uint32_t GLOBAL_GRID_SIZE = BRICK_SIZE * DEFAULT_GRID_DIM;
constexpr uint32_t ALLOC_PENDING_MARKER = 0xFFFFFFFE;
constexpr uint32_t INACTIVE = UINT32_MAX;

struct Voxel;
struct VoxelUpdate;
struct Brick;
struct BrickPool;
class BrickMap;
struct EmptyBrickInfo;

typedef uint32_t BrickIndex;
typedef uint32_t VoxelIndex;
typedef nanovdb::Coord BrickCoord;
typedef nanovdb::Coord VoxelCoord;

struct AllocationRequest {
	BrickCoord coord{};
	uint8_t valid{};
};

struct RequestCounter {
	uint32_t count;
};

struct EmptyBrickInfo {
	uint32_t gridIndex;
	bool isEmpty;
};

/// @brief Structure for a voxel.
struct Voxel {
	float density;
	float temperature;
	float fuel;
	nanovdb::Vec3f velocity;

	__hostdev__ Voxel() : density(0), temperature(0.0f), fuel(0.0f), velocity(0.0f) {}
	__hostdev__ Voxel(const float d, const float t, const float f, const nanovdb::Vec3f& v)
	    : density(d), temperature(t), fuel(f), velocity(v) {}

	__hostdev__ static bool isEmpty(const Voxel& v) {
		return v.density == 0.0f && v.temperature == 0.0f && v.fuel == 0.0f && v.velocity == nanovdb::Vec3f(0.0f);
	}
};


/// @brief Linear Array of Voxel and occupancy count.
struct Brick {
	static constexpr uint32_t VOLUME = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
	Voxel* data = nullptr;
	bool isLoaded = false;
	uint32_t poolIndex = INACTIVE;
};


struct BrickPool {
	explicit BrickPool(uint32_t maxBricks = MAX_BRICKS);
	~BrickPool();

	uint32_t allocateBrick();
	void deallocateBrick(uint32_t index);

	[[nodiscard]] __hostdev__ Brick* getDeviceBricks() const { return d_bricks; }
	[[nodiscard]] __hostdev__ Voxel* getDeviceVoxelData() const { return d_voxelData; }

   private:
	size_t m_maxBricks;
	size_t m_numAllocatedBricks;
	std::vector<uint32_t> m_freeIndices;

	Brick* d_bricks;
	Voxel* d_voxelData;
};

class BrickMap {
   public:
	explicit BrickMap(uint32_t dimX = DEFAULT_GRID_DIM, uint32_t dimY = DEFAULT_GRID_DIM, uint32_t dimZ = DEFAULT_GRID_DIM);

	~BrickMap();

	void buildFromVDB(const std::vector<std::pair<nanovdb::Coord, float>>& data) const;
	[[nodiscard]] bool allocateBrickAt(const BrickCoord& coord) const;
	void deallocateBrickAt(const BrickCoord& coord) const;
	void deallocateInactive() const;


	[[nodiscard]] uint32_t* getDeviceGrid() const { return d_grid; }
	[[nodiscard]] BrickPool* getPool() { return m_pool.get(); }

	[[nodiscard]] const BrickPool* getPool() const { return m_pool.get(); }
	[[nodiscard]] const uint32_t* getDimensions() const { return m_dimensions; }


	__host__ std::vector<nanovdb::Coord> getActiveBricks() const;
	__host__ Voxel* getBrickAtHost(uint32_t x, uint32_t y, uint32_t z) const;
	__host__ std::pair<nanovdb::Coord, nanovdb::Coord> getBrickDimensions(uint32_t x, uint32_t y, uint32_t z) const;
	__host__ void processAllocationRequests() const;

	__device__ Voxel* getBrickAtDevice(uint32_t x, uint32_t y, uint32_t z) const;
	__device__ static void requestAllocation(const BrickCoord& coord, AllocationRequest* req, RequestCounter* counter);

	__hostdev__ uint32_t getBrickIndex(const BrickCoord& coord) const {
		return coord.x() + coord.y() * m_dimensions[0] + coord.z() * m_dimensions[0] * m_dimensions[1];
	}

	__hostdev__ uint32_t getBrickIndex(const uint32_t x, const uint32_t y, const uint32_t z) const {
		return x + y * m_dimensions[0] + z * m_dimensions[0] * m_dimensions[1];
	}

	__hostdev__ Voxel* getBrickFirstVoxel(const BrickCoord& coord) const {
		const uint32_t brickIdx = getBrickIndex(coord);
		if (!isBrickActive(coord)) return nullptr;  // Check if brick is loaded
		return m_pool->getDeviceVoxelData() + brickIdx * Brick::VOLUME;
	}

	__hostdev__ Voxel* getVoxel(const BrickCoord& brickCoord, const VoxelCoord& localCoord) const {
		if (localCoord.x() >= BRICK_SIZE || localCoord.y() >= BRICK_SIZE || localCoord.z() >= BRICK_SIZE) {
			return nullptr;  // Out of bounds
		}

		uint32_t* brickidx = d_grid + getBrickIndex(brickCoord);
		if (*brickidx == UINT32_MAX) {
			return nullptr;  // Empty cell
		}

		Brick* brick = m_pool->getDeviceBricks() + *brickidx;
		if (!brick->isLoaded) {
			return nullptr;  // Brick not loaded
		}

		return brick->data + localCoord.x() + localCoord.y() * BRICK_SIZE + localCoord.z() * BRICK_SIZE * BRICK_SIZE;
	}

	__hostdev__ Voxel* getVoxel(const VoxelCoord& coord) const {
		// Check if voxel coordinates are within the global grid
		if (coord.x() >= GLOBAL_GRID_SIZE || coord.y() >= GLOBAL_GRID_SIZE || coord.z() >= GLOBAL_GRID_SIZE) {
			return nullptr;  // Out of bounds
		}
		// Convert global voxel coordinates to brick coordinates
		const uint32_t brickX = coord.x() / BRICK_SIZE;
		const uint32_t brickY = coord.y() / BRICK_SIZE;
		const uint32_t brickZ = coord.z() / BRICK_SIZE;

		const BrickCoord brickCoord(brickX, brickY, brickZ);

		// Get the first voxel of the brick
		Voxel* firstVoxel = getBrickFirstVoxel(brickCoord);
		if (firstVoxel == nullptr) return nullptr;  // Brick not loaded

		// Compute local voxel coordinates within the brick
		const uint32_t localX = coord.x() % BRICK_SIZE;
		const uint32_t localY = coord.y() % BRICK_SIZE;
		const uint32_t localZ = coord.z() % BRICK_SIZE;

		// Compute the voxel index within the brick
		const uint32_t voxelIndex = localX + localY * BRICK_SIZE + localZ * BRICK_SIZE * BRICK_SIZE;

		return firstVoxel + voxelIndex;
	}

	__hostdev__ bool isBrickActive(const BrickCoord& coord) const {
		const Brick* bricks = m_pool->getDeviceBricks();
		return bricks[getBrickIndex(coord)].isLoaded;
	}


   private:
	uint32_t m_dimensions[3];
	std::unique_ptr<BrickPool> m_pool;
	uint32_t* d_grid;

	// Computes the 1D index into the grid array from 3D coordinates.

	AllocationRequest* d_allocationRequests = nullptr;
	RequestCounter* d_requestCounter = nullptr;

	EmptyBrickInfo* d_emptyBricks = nullptr;
};
