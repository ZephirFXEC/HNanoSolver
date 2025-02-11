#pragma once

#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <nanovdb/math/Math.h>

#include <iostream>


#define CHECK_CUDA(call)                                                                                                     \
	{                                                                                                                        \
		cudaError_t err = call;                                                                                              \
		if (err != cudaSuccess) {                                                                                            \
			std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
			exit(EXIT_FAILURE);                                                                                              \
		}                                                                                                                    \
	}


constexpr int BRICK_MAP_DIM = 64;  // Top-level grid is 256^3 cells.
constexpr int BRICK_DIM = 16;       // Each brick is 32^3 voxels.
constexpr int TOTAL_BRICKS_IN_MAP = BRICK_MAP_DIM * BRICK_MAP_DIM * BRICK_MAP_DIM;
constexpr int VOXELS_PER_BRICK = BRICK_DIM * BRICK_DIM * BRICK_DIM;


struct Voxel;
struct VoxelUpdate;
struct Brick;
class BrickMap;


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
	int x{}, y{}, z{};
	Voxel voxel{};
};


/// @brief Linear Array of Voxel and occupancy count.
struct Brick {
	Voxel voxels[VOXELS_PER_BRICK]{};
	int occupancy{};
};


//-----------------------------------------------------------------------------
// Kernel to initialize the top-level brick map to –1 (meaning “no brick allocated”).
__global__ void initBrickMapKernel(int* brickMap, int totalEntries);


//-----------------------------------------------------------------------------
// Kernel to initialize the free list. The free list holds indices into the brick pool.
__global__ void initFreeListKernel(int* freeList, int* freeListTop, int maxBricks);


//-----------------------------------------------------------------------------
// Kernel to build the brick map from an array of voxel updates.
__global__ void buildBrickMapKernel(const VoxelUpdate* updates, int numUpdates, int* brickMap, Brick* bricks, int* freeList,
                                    int* freeListTop);


//-----------------------------------------------------------------------------
// A cleanup kernel that scans the grid and deallocates bricks whose occupancy is zero.
__global__ void cleanupEmptyBricksKernel(int* brickMap, Brick* bricks, int* freeList, int* freeListTop, uint32_t maxBricks);


//-----------------------------------------------------------------------------
// A kernel that calls the deviceUpdateVoxel() member function of BrickMap.
// This allows dynamic updates to be performed directly on the GPU.
__global__ void updateVoxelKernelClass(BrickMap bm, int x, int y, int z, Voxel newVoxel);


class BrickMap {
   public:
	explicit BrickMap(const uint32_t maxBricks) : _maxBricks(maxBricks) {}
	__host__ ~BrickMap() { cleanup(); }

	__hostdev__ BrickMap(const BrickMap& other)
		   : d_brickMap(other.d_brickMap)
		   , d_bricks(other.d_bricks)
		   , d_freeList(other.d_freeList)
		   , d_freeListTop(other.d_freeListTop)
		   , _maxBricks(other._maxBricks) {}

	// Host-only initialization: allocate device memory and initialize arrays.
	__host__ void initialize();

	// Host-only build: copy voxel updates to the device and launch a kernel.
	__host__ void buildFromUpdates(const std::vector<VoxelUpdate>& updates) const;

	// Host-only wrapper for a dynamic update. Launches a kernel that calls the
	__host__ void updateVoxel(int x, int y, int z, const Voxel& newVoxel) const;

	// Host-only query: copies a voxel from the device.
	__host__ Voxel queryVoxel(int x, int y, int z) const;

	// Host-only cleanup: deallocate empty bricks.
	__host__ void cleanupEmptyBricks() const;

	// Device update: update the voxel at (x,y,z) with newVoxel.
	__device__ void deviceUpdateVoxel(int x, int y, int z, const Voxel& newVoxel) const;

	// Device query: retrieve the voxel at (x,y,z) and store it in result.
	__device__ void deviceQueryVoxel(int x, int y, int z, Voxel& result) const;


   private:
	int* d_brickMap = nullptr;     // Top-level grid (size: TOTAL_BRICKS_IN_MAP)
	Brick* d_bricks = nullptr;     // Brick pool.
	int* d_freeList = nullptr;     // Free list (indices into the brick pool).
	int* d_freeListTop = nullptr;  // Count of free bricks.
	int _maxBricks;            // Maximum number of bricks in the pool.

	void cleanup() const {
		if (d_brickMap) cudaFree(d_brickMap);
		if (d_bricks) cudaFree(d_bricks);
		if (d_freeList) cudaFree(d_freeList);
		if (d_freeListTop) cudaFree(d_freeListTop);
	}

	BrickMap& operator=(const BrickMap&) = delete;

};
