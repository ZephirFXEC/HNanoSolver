//
// Created by zphrfx on 26/01/2025.
//

#pragma once

// Only define __hostdev__ when compiling as NVIDIA CUDA
#if defined(__CUDACC__) || defined(__HIP__)
#define __hostdev__ __host__ __device__
#else
#include <cmath> // for floor
#define __hostdev__
#endif

#include <nanovdb/math/Math.h>


template<template<typename> class Vec3T>
__hostdev__ inline nanovdb::Coord Floor(Vec3T<float>& xyz)
{
	const float ijk[3] = {floorf(xyz[0]), floorf(xyz[1]), floorf(xyz[2])};
	xyz[0] -= ijk[0];
	xyz[1] -= ijk[1];
	xyz[2] -= ijk[2];
	return nanovdb::Coord(int32_t(ijk[0]), int32_t(ijk[1]), int32_t(ijk[2]));
}

template <int Order>
class IndexOffsetSampler;

template <typename ValueT, int Order>
class IndexSampler;

template <>
class IndexOffsetSampler<0> {
	using AccessorT = nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex>;

   public:
	__hostdev__ explicit IndexOffsetSampler(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>& grid) : mAcc(grid) {}

	__hostdev__ [[nodiscard]] uint64_t offset(const int i, const int j, const int k) const { return mAcc.idx(i, j, k) - 1; }
	__hostdev__ [[nodiscard]] uint64_t offset(nanovdb::Coord ijk) const { return mAcc.idx(ijk[0], ijk[1], ijk[2]) - 1; }

   private:
	AccessorT mAcc;
};


template <typename ValueT>
class IndexSampler<ValueT, 0> {
   public:
	__hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler)
	    : mOffsetSampler(offsetSampler), mPos(nanovdb::Coord::max()), mOffset(0) {}

	__hostdev__ ValueT operator()(const nanovdb::Coord& ijk, const ValueT* data) const {
		if (ijk != mPos) {
			mPos = ijk;
			mOffset = mOffsetSampler.offset(ijk[0], ijk[1], ijk[2]);
		}
		return data[mOffset];
	}

   private:
	const IndexOffsetSampler<0>& mOffsetSampler;
	mutable nanovdb::Coord mPos;
	mutable uint64_t mOffset;
};


template <typename ValueT>
class TrilinearSampler {
   public:
	__hostdev__ explicit TrilinearSampler(const IndexOffsetSampler<0>& offsetSampler) : mOffsetSampler(offsetSampler) {}

	[[nodiscard]] const IndexOffsetSampler<0>& Acc() const { return mOffsetSampler; }

	void stencil(nanovdb::Coord& ijk, ValueT (&v)[2][2][2], const ValueT* data) const {
		// (i,   j,   k)
		v[0][0][0] = data[mOffsetSampler.offset(ijk)];

		// (i,   j,   k+1)
		ijk[2] += 1;
		v[0][0][1] = data[mOffsetSampler.offset(ijk)];

		// (i,   j+1, k+1)
		ijk[1] += 1;
		v[0][1][1] = data[mOffsetSampler.offset(ijk)];

		// (i,   j+1, k)
		ijk[2] -= 1;
		v[0][1][0] = data[mOffsetSampler.offset(ijk)];

		// (i+1, j,   k)
		ijk[0] += 1;
		ijk[1] -= 1;
		v[1][0][0] = data[mOffsetSampler.offset(ijk)];

		// (i+1, j,   k+1)
		ijk[2] += 1;
		v[1][0][1] = data[mOffsetSampler.offset(ijk)];

		// (i+1, j+1, k+1)
		ijk[1] += 1;
		v[1][1][1] = data[mOffsetSampler.offset(ijk)];

		// (i+1, j+1, k)
		ijk[2] -= 1;
		v[1][1][0] = data[mOffsetSampler.offset(ijk)];
	}

	template <typename RealT, template <typename...> class Vec3T>
	static __hostdev__ ValueT sample(const Vec3T<RealT>& uvw, const ValueT (&v)[2][2][2]) {
		auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };
		return lerp(lerp(lerp(v[0][0][0], v[0][0][1], uvw[2]), lerp(v[0][1][0], v[0][1][1], uvw[2]), uvw[1]),
		            lerp(lerp(v[1][0][0], v[1][0][1], uvw[2]), lerp(v[1][1][0], v[1][1][1], uvw[2]), uvw[1]), uvw[0]);
	}

   private:
	const IndexOffsetSampler<0>& mOffsetSampler;
};


template <typename ValueT>
class IndexSampler<ValueT, 1> : public TrilinearSampler<ValueT> {
	using BaseT = TrilinearSampler<ValueT>;

   public:
	__hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler)
	    : BaseT(offsetSampler), mPos(nanovdb::Coord::max()), mValues() {}


	template <typename RealT, template <typename...> class Vec3T>
	__hostdev__ ValueT operator()(Vec3T<RealT> xyz, const ValueT* data) const {

		this->cache(xyz, data);
		return BaseT::sample(xyz, mValues);
	}

	__hostdev__ ValueT operator()(const nanovdb::Coord& ijk, const ValueT* data) const {
		return ijk == mPos ? mValues[0][0][0] : data[BaseT::Acc().offset(ijk)];
	}


   private:
	mutable nanovdb::Coord mPos;
	mutable ValueT mValues[2][2][2];

	template <typename RealT, template <typename...> class Vec3T>
	__hostdev__ void cache(Vec3T<RealT>& xyz, const ValueT* data) const {
		if (nanovdb::Coord ijk = Floor(xyz); ijk != mPos) {
			mPos = ijk;
			BaseT::stencil(ijk, mValues, data);
		}
	}
};