#pragma once

#include "BrickMap.cuh"

// Only define __hostdev__ when compiling as NVIDIA CUDA
#if defined(__CUDACC__) || defined(__HIP__)
#define __hostdev__ __host__ __device__
#else
#include <cmath>  // for floor
#define __hostdev__
#endif

#include "nanovdb/math/Math.h"


template <template <typename> class Vec3T>
__hostdev__ inline nanovdb::Coord Floor(Vec3T<float>& xyz) {
	const float ijk[3] = {floorf(xyz[0]), floorf(xyz[1]), floorf(xyz[2])};
	xyz[0] -= ijk[0];
	xyz[1] -= ijk[1];
	xyz[2] -= ijk[2];
	return nanovdb::Coord(int32_t(ijk[0]), int32_t(ijk[1]), int32_t(ijk[2]));
}


__device__ inline Voxel sampleVoxel(const uint32_t* grid, const Brick* bricks, const Dim dim, const VoxelCoordGlobal& globalCoord) {
	// Convert to brick coordinates
	const BrickCoord brickCoord = BrickMath::getBrickCoord(globalCoord);

	// Get brick index from grid
	const uint32_t gridIdx = BrickMath::getBrickIndex(brickCoord, dim);
	const uint32_t brickIdx = grid[gridIdx];

	if (brickIdx == BrickConfig::INACTIVE) return {};  // Brick not active

	const auto& [data, isLoaded] = bricks[brickIdx];

	if (!isLoaded) return {};  // Brick not loaded

	const VoxelCoordLocal localCoord = BrickMath::globalToLocalCoord(globalCoord);
	const uint32_t voxelIdx = BrickMath::localToVoxelIdx(localCoord);
	const Voxel* brickData = data;

	return brickData[voxelIdx];
}

__device__ Voxel trilinearInterpolate(float wx, float wy, float wz, const Voxel& v000, const Voxel& v001, const Voxel& v010,
                                      const Voxel& v011, const Voxel& v100, const Voxel& v101, const Voxel& v110, const Voxel& v111) {
	Voxel result;
	const float omwx = 1.0f - wx;
	const float omwy = 1.0f - wy;
	const float omwz = 1.0f - wz;

	// Precompute weights.
	const float w000 = omwx * omwy * omwz;
	const float w001 = omwx * omwy * wz;
	const float w010 = omwx * wy * omwz;
	const float w011 = omwx * wy * wz;
	const float w100 = wx * omwy * omwz;
	const float w101 = wx * omwy * wz;
	const float w110 = wx * wy * omwz;
	const float w111 = wx * wy * wz;

	// Use nested __fmaf_rn calls to accumulate density.
	result.density = __fmaf_rn(
	    w111, v111.density,
	    __fmaf_rn(w110, v110.density,
	              __fmaf_rn(w101, v101.density,
	                        __fmaf_rn(w100, v100.density,
	                                  __fmaf_rn(w011, v011.density,
	                                            __fmaf_rn(w010, v010.density, __fmaf_rn(w001, v001.density, w000 * v000.density)))))));

	// Compute velocity per component.
	result.velocity[0] =
	    __fmaf_rn(w111, v111.velocity[0],
	              __fmaf_rn(w110, v110.velocity[0],
	                        __fmaf_rn(w101, v101.velocity[0],
	                                  __fmaf_rn(w100, v100.velocity[0],
	                                            __fmaf_rn(w011, v011.velocity[0],
	                                                      __fmaf_rn(w010, v010.velocity[0],
	                                                                __fmaf_rn(w001, v001.velocity[0], w000 * v000.velocity[0])))))));

	result.velocity[1] =
	    __fmaf_rn(w111, v111.velocity[1],
	              __fmaf_rn(w110, v110.velocity[1],
	                        __fmaf_rn(w101, v101.velocity[1],
	                                  __fmaf_rn(w100, v100.velocity[1],
	                                            __fmaf_rn(w011, v011.velocity[1],
	                                                      __fmaf_rn(w010, v010.velocity[1],
	                                                                __fmaf_rn(w001, v001.velocity[1], w000 * v000.velocity[1])))))));

	result.velocity[2] =
	    __fmaf_rn(w111, v111.velocity[2],
	              __fmaf_rn(w110, v110.velocity[2],
	                        __fmaf_rn(w101, v101.velocity[2],
	                                  __fmaf_rn(w100, v100.velocity[2],
	                                            __fmaf_rn(w011, v011.velocity[2],
	                                                      __fmaf_rn(w010, v010.velocity[2],
	                                                                __fmaf_rn(w001, v001.velocity[2], w000 * v000.velocity[2])))))));

	return result;
}


__device__ inline Voxel trilinearSample(const uint32_t* grid, const Brick* bricks, const Dim dim, const nanovdb::Vec3f xyz) {
	const int x0 = __float2int_rd(xyz[0]), x1 = x0 + 1;
	const int y0 = __float2int_rd(xyz[1]), y1 = y0 + 1;
	const int z0 = __float2int_rd(xyz[2]), z1 = z0 + 1;

	// Calculate interpolation weights
	const float wx = xyz[0] - x0;
	const float wy = xyz[1] - y0;
	const float wz = xyz[2] - z0;

	// Sample all eight corners
	const Voxel& v000 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x0, y0, z0));
	const Voxel& v001 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x0, y0, z1));
	const Voxel& v010 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x0, y1, z0));
	const Voxel& v011 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x0, y1, z1));
	const Voxel& v100 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x1, y0, z0));
	const Voxel& v101 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x1, y0, z1));
	const Voxel& v110 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x1, y1, z0));
	const Voxel& v111 = sampleVoxel(grid, bricks, dim, VoxelCoordGlobal(x1, y1, z1));

	return trilinearInterpolate(wx, wy, wz, v000, v001, v010, v011, v100, v101, v110, v111);
}

/*

template <int Order>
class IndexOffsetSampler;

template <typename ValueT, int Order, bool Cache = true>
class IndexSampler;

template <>
class IndexOffsetSampler<0> {
   public:
    __hostdev__ explicit IndexOffsetSampler(const BrickMap* const brickMap) : mBrickMap(brickMap) {}

    __hostdev__ [[nodiscard]] uint32_t getBrickIdx(const uint32_t i, const uint32_t j, const uint32_t k) const {
        return BrickMath::getBrickIndex(i, j, k, mBrickMap->getDimensions());
    }
    __hostdev__ [[nodiscard]] uint32_t getBrickIdx(const BrickCoord& ijk) const {
        return BrickMath::getBrickIndex(ijk, mBrickMap->getDimensions());
    }

    __hostdev__ [[nodiscard]] bool isBrickActive(const BrickCoord& ijk) const { return mBrickMap->isBrickActive(ijk); }

    __hostdev__ [[nodiscard]] uint32_t getVoxelIdx(const VoxelCoordLocal& local) const { return BrickMath::localToVoxelIdx(local); }

    __hostdev__ [[nodiscard]] uint32_t getVoxelIdxGlobal(const VoxelCoordGlobal& global) const {
        return BrickMath::localToVoxelIdx(BrickMath::globalToLocalCoord(global));
    }

   private:
    const BrickMap* const mBrickMap;
};


template <typename ValueT>
class IndexSampler<ValueT, 0> {
   public:
    __hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data)
        : mOffsetSampler(offsetSampler), mPos(nanovdb::Coord::max()), mOffset(0), mIsActive(false), mData(data) {}

    __hostdev__ bool isDataActive(const nanovdb::Coord& ijk) const {
        updateCache(ijk);
        return mIsActive && mData[mOffset] != ValueT(0);
    }

    __hostdev__ ValueT operator()(const nanovdb::Coord& ijk) const {
        updateCache(ijk);
        if (!mIsActive) {
            return ValueT(0);
        }
        return mData[mOffset];
    }

   private:
    __hostdev__ void updateCache(const nanovdb::Coord& ijk) const {
        if (ijk != mPos) {
            mPos = ijk;
            mIsActive = mOffsetSampler.isBrickActive(ijk);
            if (mIsActive) {
                mOffset = mOffsetSampler.pos(ijk);
            }
        }
    }

    const IndexOffsetSampler<0>& mOffsetSampler;
    mutable nanovdb::Coord mPos;
    mutable uint32_t mOffset;
    mutable bool mIsActive;
    const ValueT* mData;
};


template <typename ValueT>
class TrilinearSampler {
   public:
    __hostdev__ explicit TrilinearSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data)
        : mNearestSampler(offsetSampler, data) {}

    [[nodiscard]] __hostdev__ const IndexSampler<ValueT, 0>& Acc() const { return mNearestSampler; }

    __hostdev__ void stencil(const nanovdb::Coord& ijk, ValueT (&v)[2][2][2]) const {
        // Local copy to preserve original coordinates.
        nanovdb::Coord tmp = ijk;
        auto getVal = [&](int di, int dj, int dk) -> ValueT {
            const nanovdb::Coord coord(tmp[0] + di, tmp[1] + dj, tmp[2] + dk);
            return mNearestSampler(coord);
        };

        // ZYX order traversal (v[x][y][z])
        v[0][0][0] = getVal(0, 0, 0);  // (i, j, k)
        v[0][0][1] = getVal(0, 0, 1);  // (i, j, k+1)
        v[0][1][0] = getVal(0, 1, 0);  // (i, j+1, k)
        v[0][1][1] = getVal(0, 1, 1);  // (i, j+1, k+1)
        v[1][0][0] = getVal(1, 0, 0);  // (i+1, j, k)
        v[1][0][1] = getVal(1, 0, 1);  // (i+1, j, k+1)
        v[1][1][0] = getVal(1, 1, 0);  // (i+1, j+1, k)
        v[1][1][1] = getVal(1, 1, 1);  // (i+1, j+1, k+1)
    }

    template <typename RealT, template <typename...> class Vec3T>
    static __hostdev__ ValueT sample(const Vec3T<RealT>& uvw, const ValueT (&v)[2][2][2]) {
        auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };

        const ValueT z0 = lerp(v[0][0][0], v[0][0][1], uvw[2]);
        const ValueT z1 = lerp(v[0][1][0], v[0][1][1], uvw[2]);
        const ValueT z2 = lerp(v[1][0][0], v[1][0][1], uvw[2]);
        const ValueT z3 = lerp(v[1][1][0], v[1][1][1], uvw[2]);

        const ValueT y0 = lerp(z0, z1, uvw[1]);
        const ValueT y1 = lerp(z2, z3, uvw[1]);

        return lerp(y0, y1, uvw[0]);
    }

   private:
    const IndexSampler<ValueT, 0> mNearestSampler;
};


template <typename ValueT>
class IndexSampler<ValueT, 1> : public TrilinearSampler<ValueT> {
    using BaseT = TrilinearSampler<ValueT>;

   public:
    __hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data)
        : BaseT(offsetSampler, data), mPos(nanovdb::Coord::max()), mValues{} {}

    __hostdev__ bool isDataActive(const nanovdb::Coord& ijk) const { return BaseT::Acc().isDataActive(ijk); }

    template <typename RealT, template <typename...> class Vec3T>
    __hostdev__ ValueT operator()(Vec3T<RealT> xyz) const {
        this->cache(xyz);
        return BaseT::sample(xyz, mValues);
    }

    __hostdev__ ValueT operator()(const nanovdb::Coord& ijk) const {
        if (!BaseT::Acc().isDataActive(ijk)) {
            return ValueT(0);
        }
        return (ijk == mPos) ? mValues[0][0][0] : BaseT::Acc()(ijk);
    }

   private:
    mutable nanovdb::Coord mPos;
    mutable ValueT mValues[2][2][2];

    template <typename RealT, template <typename...> class Vec3T>
    __hostdev__ void cache(Vec3T<RealT>& xyz) const {
        // Compute the lower-bound (integer) coordinate.
        nanovdb::Coord ijk = Floor(xyz);
        if (ijk != mPos) {
            mPos = ijk;
            BaseT::stencil(ijk, mValues);
        }
    }
};

*/