//
// Created by zphrfx on 26/01/2025.
//

#pragma once

// Only define __hostdev__ when compiling as NVIDIA CUDA
#if defined(__CUDACC__) || defined(__HIP__)
#define __hostdev__ __host__ __device__
#else
#include <cmath>  // for floor
#define __hostdev__
#endif

#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/Math.h"


template <template <typename> class Vec3T>
__hostdev__ inline nanovdb::Coord Floor(Vec3T<float>& xyz) {
	const float ijk[3] = {floorf(xyz[0]), floorf(xyz[1]), floorf(xyz[2])};
	xyz[0] -= ijk[0];
	xyz[1] -= ijk[1];
	xyz[2] -= ijk[2];
	return nanovdb::Coord(int32_t(ijk[0]), int32_t(ijk[1]), int32_t(ijk[2]));
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

	__hostdev__ [[nodiscard]] uint64_t offset(const int i, const int j, const int k) const { return mAcc.idx(i, j, k) - 1; }
	__hostdev__ [[nodiscard]] uint64_t offset(const nanovdb::Coord ijk) const { return mAcc.getIndex(ijk) - 1; }
	__hostdev__ [[nodiscard]] bool isActive(const int i, const int j, const int k) const { return mAcc.isActive(nanovdb::Coord(i, j, k)); }
	__hostdev__ [[nodiscard]] bool isActive(const nanovdb::Coord ijk) const { return mAcc.isActive(ijk); }

   private:
	AccessorT mAcc;
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