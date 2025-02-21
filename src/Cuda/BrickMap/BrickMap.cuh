#pragma once

#include <cuda_runtime.h>
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


namespace BrickConfig {
constexpr uint32_t BRICK_SIZE = 32;
constexpr uint32_t DEFAULT_GRID_DIM = 256;
constexpr size_t MAX_BRICKS = 128 * 128;
constexpr size_t MAX_ALLOCATION_REQUESTS = MAX_BRICKS;
constexpr uint32_t GLOBAL_GRID_SIZE = BRICK_SIZE * DEFAULT_GRID_DIM;
enum BrickStatus : uint32_t { INACTIVE = UINT32_MAX, ALLOC_PENDING = 0xFFFFFFFE };
}  // namespace BrickConfig

// Forward declarations
struct Voxel;
struct Brick;
class BrickPool;
class BrickMap;

using BrickIndex = uint32_t;
using VoxelIndex = uint32_t;
using Dim = nanovdb::math::Vec3<uint8_t>;
using BrickCoord = nanovdb::math::Vec3<uint8_t>;
using VoxelCoordLocal = nanovdb::math::Vec3<uint16_t>;
using VoxelCoordGlobal = nanovdb::math::Vec3<uint32_t>;


namespace BrickMath {

__hostdev__ inline uint32_t getBrickIndex(const BrickCoord& coord, const Dim dims) {
	return coord[0] + coord[1] * dims[0] + coord[2] * dims[0] * dims[1];
}
__hostdev__ inline BrickCoord getBrickGridIdxToCoord(const uint32_t idx, const Dim dims) {
	return BrickCoord(idx % dims[0], (idx / dims[0]) % dims[1], idx / (dims[0] * dims[1]));
}

__hostdev__ inline bool isValidGlobalVoxelCoord(const VoxelCoordGlobal& global) {
	return global[0] < BrickConfig::GLOBAL_GRID_SIZE && global[1] < BrickConfig::GLOBAL_GRID_SIZE &&
	       global[2] < BrickConfig::GLOBAL_GRID_SIZE;
}

__hostdev__ inline VoxelCoordLocal globalToLocalCoord(const VoxelCoordGlobal& global) {
	return VoxelCoordLocal(global[0] % BrickConfig::BRICK_SIZE, global[1] % BrickConfig::BRICK_SIZE, global[2] % BrickConfig::BRICK_SIZE);
}

__hostdev__ inline VoxelCoordGlobal localToGlobalCoord(const BrickCoord& brick, const VoxelCoordLocal& local) {
	return VoxelCoordGlobal(brick[0] * BrickConfig::BRICK_SIZE + local[0], brick[1] * BrickConfig::BRICK_SIZE + local[1],
	                        brick[2] * BrickConfig::BRICK_SIZE + local[2]);
}

__hostdev__ inline VoxelCoordGlobal localToGlobalCoord(const uint32_t brickidx, const VoxelCoordLocal& local) {
	return localToGlobalCoord(getBrickGridIdxToCoord(brickidx, Dim(BrickConfig::DEFAULT_GRID_DIM)), local);
}

__hostdev__ inline uint32_t localToVoxelIdx(const VoxelCoordLocal& local) {
	return local[0] + local[1] * BrickConfig::BRICK_SIZE + local[2] * BrickConfig::BRICK_SIZE * BrickConfig::BRICK_SIZE;
}

__hostdev__ inline VoxelCoordLocal voxelIdxToLocal(const uint32_t idx) {
	return VoxelCoordLocal(idx % BrickConfig::BRICK_SIZE, (idx / BrickConfig::BRICK_SIZE) % BrickConfig::BRICK_SIZE,
	                       idx / (BrickConfig::BRICK_SIZE * BrickConfig::BRICK_SIZE));
}

__hostdev__ inline BrickCoord getBrickCoord(const VoxelCoordGlobal& global) {
	return BrickCoord(global[0] / BrickConfig::BRICK_SIZE, global[1] / BrickConfig::BRICK_SIZE, global[2] / BrickConfig::BRICK_SIZE);
}

__hostdev__ inline bool isValidLocalCoord(const VoxelCoordLocal& local) {
	return local[0] < BrickConfig::BRICK_SIZE && local[1] < BrickConfig::BRICK_SIZE && local[2] < BrickConfig::BRICK_SIZE;
}


__hostdev__ inline bool isValidBrickCoord(const BrickCoord& coord, const Dim dims) {
	return coord[0] < dims[0] && coord[1] < dims[1] && coord[2] < dims[2];
}


}  // namespace BrickMath

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

struct Brick {
	static constexpr uint32_t VOLUME = BrickConfig::BRICK_SIZE * BrickConfig::BRICK_SIZE * BrickConfig::BRICK_SIZE;
	/// @brief Array of voxels in brick from [0 to 32^2 - 1].
	Voxel* data = nullptr;
	bool isLoaded = false;
};


class BrickPool {
   public:
	explicit BrickPool(uint32_t maxBricks);
	~BrickPool();

	uint32_t allocateBrick();
	void deallocateBrick(uint32_t index);

	__hostdev__ Brick* getDeviceBricks() const { return d_bricks; }
	__hostdev__ Voxel* getDeviceVoxelData() const { return d_voxelData; }
	__hostdev__ uint32_t getNumAllocatedBricks() const { return m_numAllocatedBricks; }

   private:
	size_t m_maxBricks;
	size_t m_numAllocatedBricks;
	std::vector<uint32_t> m_freeIndices;

	Brick* d_bricks;
	Voxel* d_voxelData;
};

class BrickMap {
   public:
	explicit BrickMap(Dim dim = Dim(BrickConfig::DEFAULT_GRID_DIM));
	~BrickMap();

	void buildFromVDB(const std::vector<std::pair<nanovdb::Coord, float>>& data) const;
	[[nodiscard]] bool allocateBrickAt(const BrickCoord& coord) const;
	void deallocateBrickAt(const BrickCoord& coord) const;
	void deallocateInactive() const;

	[[nodiscard]] __hostdev__ uint32_t* getDeviceGrid() const { return d_grid; }
	[[nodiscard]] __hostdev__ BrickPool* getPool() { return m_pool; }
	[[nodiscard]] __hostdev__ const BrickPool* getPool() const { return m_pool; }
	[[nodiscard]] __hostdev__ Dim getDimensions() const { return m_dimensions; }

	__host__ std::vector<BrickCoord> getActiveBricks() const;
	__host__ Voxel* getBrickAtHost(const BrickCoord& coord) const;

	__host__ void processAllocationRequests() const;

	__device__ Voxel* getBrickAtDevice(const BrickCoord& coord) const;
	__device__ static void requestAllocation(const BrickCoord& coord, AllocationRequest* req, RequestCounter* counter);

	AllocationRequest* getDeviceAllocationRequests() const { return d_allocationRequests; }
	RequestCounter* getDeviceRequestCounter() const { return d_requestCounter; }

   private:
	Dim m_dimensions;
	BrickPool* m_pool;
	uint32_t* d_grid;

	AllocationRequest* d_allocationRequests = nullptr;
	RequestCounter* d_requestCounter = nullptr;
	EmptyBrickInfo* d_emptyBricks = nullptr;
};



