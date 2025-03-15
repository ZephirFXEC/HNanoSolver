//
// Created by zphrfx on 26/01/2025.
//

#pragma once

// Only define __hostdev__ when compiling as NVIDIA CUDA
#if defined(__CUDACC__) || defined(__HIP__)
#define __hostdev__ __host__ __device__
#include <cuda_runtime.h>  // for __float2int_rd
#else
#include <cmath>  // for floor
#define __hostdev__
#endif

#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/Math.h"

// Scalar fmaf wrapper that handles both CPU and GPU
inline __hostdev__ float fmaf_scalar(const float a, const float b, const float c) {
#if defined(__CUDA_ARCH__)
	return __fmaf_rn(a, b, c);
#else
	return std::fmaf(a, b, c);  // Use std::fmaf for CPU-only compilation
#endif
}

// Fused multiply-add for vec3
inline __device__ nanovdb::Vec3f fmaf(const float a, const nanovdb::Vec3f& b, const nanovdb::Vec3f& c) {
	return {fmaf_scalar(a, b[0], c[0]), fmaf_scalar(a, b[1], c[1]), fmaf_scalar(a, b[2], c[2])};
}

inline __host__ openvdb::Vec3f fmaf(const float a, const openvdb::Vec3f& b, const openvdb::Vec3f& c) {
	return {fmaf_scalar(a, b[0], c[0]), fmaf_scalar(a, b[1], c[1]), fmaf_scalar(a, b[2], c[2])};
}

template <template <typename> class Vec3T>
__hostdev__ inline nanovdb::Coord Floor(Vec3T<float>& xyz) {
#ifdef __CUDA_ARCH__
	const int32_t i = __float2int_rd(xyz[0]);
	const int32_t j = __float2int_rd(xyz[1]);
	const int32_t k = __float2int_rd(xyz[2]);
#else
	const int32_t i = static_cast<int32_t>(std::floor(xyz[0]));
	const int32_t j = static_cast<int32_t>(std::floor(xyz[1]));
	const int32_t k = static_cast<int32_t>(std::floor(xyz[2]));
#endif

	// Compute fractional parts in a more optimized way
	xyz[0] -= static_cast<float>(i);
	xyz[1] -= static_cast<float>(j);
	xyz[2] -= static_cast<float>(k);

	return nanovdb::Coord(i, j, k);
}

template <int Order>
class IndexOffsetSampler;

template <typename ValueT, int Order, bool Cache = true>
class IndexSampler;

template <>
class IndexOffsetSampler<0> {
	using AccessorT = nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex>;

   public:
	__hostdev__ explicit IndexOffsetSampler(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>& grid) : mAcc(grid) {}

	__hostdev__ __forceinline__ uint64_t offset(const int i, const int j, const int k) const { return mAcc.idx(i, j, k) - 1; }

	__hostdev__ __forceinline__ uint64_t offset(const nanovdb::Coord ijk) const { return mAcc.getIndex(ijk) - 1; }

	__hostdev__ __forceinline__ bool isActive(const int i, const int j, const int k) const {
		return mAcc.isActive(nanovdb::Coord(i, j, k));
	}

	__hostdev__ __forceinline__ bool isActive(const nanovdb::Coord ijk) const { return mAcc.isActive(ijk); }

   private:
	AccessorT mAcc;
};


template <typename ValueT>
class IndexSampler<ValueT, 0> {
   public:
	__hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data)
	    : mOffsetSampler(offsetSampler), mPos(nanovdb::Coord::max()), mOffset(0), mIsActive(false), mData(data) {}

	__hostdev__ __forceinline__ bool isDataActive(const nanovdb::Coord& ijk) const {
		updateCache(ijk);
		return mIsActive && mData[mOffset] != ValueT(0);
	}

	__hostdev__ __forceinline__ ValueT operator()(const nanovdb::Coord& ijk) const {
		updateCache(ijk);
		return mIsActive ? mData[mOffset] : ValueT(0);
	}

   private:
	__hostdev__ void updateCache(const nanovdb::Coord& ijk) const {
		if (ijk != mPos) {
			mPos = ijk;
			mIsActive = mOffsetSampler.isActive(ijk);
			if (mIsActive) {
				mOffset = mOffsetSampler.offset(ijk);
			}
		}
	}

	const IndexOffsetSampler<0>& mOffsetSampler;
	mutable nanovdb::Coord mPos;
	mutable uint64_t mOffset;
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

		// ZYX order traversal (v[x][y][z])
		// Pre-compute common offsets to reduce coordinate calculations
		const nanovdb::Coord c000(tmp[0], tmp[1], tmp[2]);
		const nanovdb::Coord c001(tmp[0], tmp[1], tmp[2] + 1);
		const nanovdb::Coord c010(tmp[0], tmp[1] + 1, tmp[2]);
		const nanovdb::Coord c011(tmp[0], tmp[1] + 1, tmp[2] + 1);
		const nanovdb::Coord c100(tmp[0] + 1, tmp[1], tmp[2]);
		const nanovdb::Coord c101(tmp[0] + 1, tmp[1], tmp[2] + 1);
		const nanovdb::Coord c110(tmp[0] + 1, tmp[1] + 1, tmp[2]);
		const nanovdb::Coord c111(tmp[0] + 1, tmp[1] + 1, tmp[2] + 1);


		// Gather values with direct coordinate access
		v[0][0][0] = mNearestSampler(c000);
		v[0][0][1] = mNearestSampler(c001);
		v[0][1][0] = mNearestSampler(c010);
		v[0][1][1] = mNearestSampler(c011);
		v[1][0][0] = mNearestSampler(c100);
		v[1][0][1] = mNearestSampler(c101);
		v[1][1][0] = mNearestSampler(c110);
		v[1][1][1] = mNearestSampler(c111);
	}

	template <typename RealT, template <typename...> class Vec3T>
	static __hostdev__ ValueT sample(const Vec3T<RealT>& uvw, const ValueT (&stencil)[2][2][2]) {
		const RealT u = uvw[0];
		const RealT v = uvw[1];
		const RealT w = uvw[2];

		// Compute complementary weights
		const RealT u1 = RealT(1) - u;
		const RealT v1 = RealT(1) - v;
		const RealT w1 = RealT(1) - w;


		// First interpolate along Z
		const ValueT z00 = fmaf(w, stencil[0][0][1], w1 * stencil[0][0][0]);
		const ValueT z01 = fmaf(w, stencil[0][1][1], w1 * stencil[0][1][0]);
		const ValueT z10 = fmaf(w, stencil[1][0][1], w1 * stencil[1][0][0]);
		const ValueT z11 = fmaf(w, stencil[1][1][1], w1 * stencil[1][1][0]);

		// Then along Y
		const ValueT y0 = fmaf(v, z01, v1 * z00);
		const ValueT y1 = fmaf(v, z11, v1 * z10);

		// Finally along X
		return fmaf(u, y1, u1 * y0);
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
		// Fast path for cached position
		if (ijk == mPos) return mValues[0][0][0];

		// Avoid unnecessary memory access if not active
		if (!BaseT::Acc().isDataActive(ijk)) {
			return ValueT(0);
		}

		return BaseT::Acc()(ijk);
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