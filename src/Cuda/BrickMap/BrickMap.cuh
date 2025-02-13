#pragma once

#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <nanovdb/math/Math.h>

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
constexpr size_t MAX_BRICKS = 128 * 128;              // Adjustable based on GPU memory
constexpr size_t MAX_ALLOCATION_REQUESTS = MAX_BRICKS;  // Maximum pending allocations

struct Voxel;
struct VoxelUpdate;
struct Brick;
struct BrickPool;
class BrickMap;


struct BrickMapStats {
	uint32_t allocatedBricks{0};
	uint32_t activeVoxels{0};
	float memoryUsageMB{0.0f};
	float occupancyRatio{0.0f};  // activeVoxels / (allocatedBricks * VOXELS_PER_BRICK)
};


struct AllocationRequest {
	uint32_t x, y, z;
	bool valid;
};

struct RequestCounter {
	uint32_t count;
};

/// @brief Structure for a voxel.
struct Voxel {
	float density;
	float temperature;
	float fuel;
	nanovdb::Vec3f velocity;

	__hostdev__ Voxel() : density(0.0f), temperature(0.0f), fuel(0.0f), velocity(0.0f) {}
	__hostdev__ Voxel(const float d, const float t, const float f, const nanovdb::Vec3f& v)
	    : density(d), temperature(t), fuel(f), velocity(v) {}

	__hostdev__ static bool isEmpty(const Voxel& v) {
		return v.density == 0.0f && v.temperature == 0.0f && v.fuel == 0.0f && v.velocity == nanovdb::Vec3f(0.0f);
	}
};


/// @brief Structure for a voxel update.
/// Contain a Voxel and its global coordinates.
struct VoxelUpdate {
	nanovdb::Coord coord{};
	Voxel voxel{};
};


/// @brief Linear Array of Voxel and occupancy count.
struct Brick {
	static constexpr uint32_t VOLUME = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
	Voxel* data = nullptr;            // Device pointer to voxel data for this brick
	bool isLoaded = false;            // Whether this brick is loaded (active) on the GPU
	uint32_t poolIndex = UINT32_MAX;  // The index in the BrickPool (used for bookkeeping)
};


struct BrickPool {
	explicit BrickPool(uint32_t maxBricks = MAX_BRICKS);
	~BrickPool();

	uint32_t allocateBrick();
	void deallocateBrick(uint32_t index);

	[[nodiscard]] Brick* getDeviceBricks() const { return d_bricks; }
	[[nodiscard]] Voxel* getDeviceVoxelData() const { return d_voxelData; }

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

	[[nodiscard]] bool allocateBrickAt(uint32_t x, uint32_t y, uint32_t z) const;
	void deallocateBrickAt(uint32_t x, uint32_t y, uint32_t z) const;


	uint32_t* getDeviceGrid() const { return d_grid; }
	BrickPool* getPool() { return m_pool.get(); }

	[[nodiscard]] const BrickPool* getPool() const { return m_pool.get(); }
	[[nodiscard]] const uint32_t* getDimensions() const { return m_dimensions; }
	[[nodiscard]] Voxel* getVoxelAt(uint32_t x, uint32_t y, uint32_t z) const;

	__device__ void requestAllocation(uint32_t x, uint32_t y, uint32_t z,AllocationRequest* req, RequestCounter* counter);
	__host__ void processAllocationRequests();

private:
	uint32_t m_dimensions[3];
	std::unique_ptr<BrickPool> m_pool;
	uint32_t* d_grid;

	// Computes the 1D index into the grid array from 3D coordinates.
	[[nodiscard]] uint32_t getGridIndex(const uint32_t x, const uint32_t y, const uint32_t z) const {
		return x + y * m_dimensions[0] + z * m_dimensions[0] * m_dimensions[1];
	}

	AllocationRequest* d_allocationRequests = nullptr;
	RequestCounter* d_requestCounter = nullptr;
};
