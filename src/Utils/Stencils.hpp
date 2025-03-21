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

// Fused multiply-add for vec3
inline __device__ nanovdb::Vec3f fmaf(const float a, const nanovdb::Vec3f& b, const nanovdb::Vec3f& c) {
	return {__fmaf_rn(a, b[0], c[0]), __fmaf_rn(a, b[1], c[1]), __fmaf_rn(a, b[2], c[2])};
}


template <template <typename> class Vec3T>
__hostdev__ inline nanovdb::Coord Floor(Vec3T<float>& xyz) {
#ifdef __CUDACC__
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
	    : mOffsetSampler(offsetSampler), mData(data) {}


	__hostdev__ __forceinline__ ValueT operator()(const nanovdb::Coord& ijk) const {
		return mOffsetSampler.isActive(ijk) ? mData[mOffsetSampler.offset(ijk)] : ValueT(0);
	}

	const IndexOffsetSampler<0>& mOffsetSampler;
	const ValueT* mData;
};


template <typename ValueT>
class TrilinearSampler {
   public:
	__hostdev__ explicit TrilinearSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data)
	    : mNearestSampler(offsetSampler, data) {}

	[[nodiscard]] __hostdev__ const IndexSampler<ValueT, 0>& Acc() const { return mNearestSampler; }

	__hostdev__ void stencil(const nanovdb::Coord& ijk, ValueT (&v)[2][2][2]) const {
		// ZYX order traversal (v[x][y][z])
		const nanovdb::Coord c000(ijk[0], ijk[1], ijk[2]);
		const nanovdb::Coord c001(ijk[0], ijk[1], ijk[2] + 1);
		const nanovdb::Coord c010(ijk[0], ijk[1] + 1, ijk[2]);
		const nanovdb::Coord c011(ijk[0], ijk[1] + 1, ijk[2] + 1);
		const nanovdb::Coord c100(ijk[0] + 1, ijk[1], ijk[2]);
		const nanovdb::Coord c101(ijk[0] + 1, ijk[1], ijk[2] + 1);
		const nanovdb::Coord c110(ijk[0] + 1, ijk[1] + 1, ijk[2]);
		const nanovdb::Coord c111(ijk[0] + 1, ijk[1] + 1, ijk[2] + 1);

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

	// Changed sample from static to non-static so that it can call the stencil member function.
	template <typename RealT, template <typename...> class Vec3T>
	__hostdev__ ValueT sample(const Vec3T<RealT>& uvw) const {
		// Make a copy to compute fractional parts (and preserve uvw)
		auto tmp = uvw;
		const nanovdb::Coord ijk = Floor(tmp);

		ValueT v[2][2][2];
		this->stencil(ijk, v);


		auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };

		const ValueT z0 = lerp(v[0][0][0], v[0][0][1], tmp[2]);
		const ValueT z1 = lerp(v[0][1][0], v[0][1][1], tmp[2]);
		const ValueT z2 = lerp(v[1][0][0], v[1][0][1], tmp[2]);
		const ValueT z3 = lerp(v[1][1][0], v[1][1][1], tmp[2]);

		const ValueT y0 = lerp(z0, z1, tmp[1]);
		const ValueT y1 = lerp(z2, z3, tmp[1]);

		return lerp(y0, y1, tmp[0]);
	}

   private:
	const IndexSampler<ValueT, 0> mNearestSampler;
};


template <typename ValueT>
class IndexSampler<ValueT, 1> : public TrilinearSampler<ValueT> {
	using BaseT = TrilinearSampler<ValueT>;

   public:
	__hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data) : BaseT(offsetSampler, data) {}

	__hostdev__ bool isDataActive(const nanovdb::Coord& ijk) const { return BaseT::Acc().mOffsetSampler.isActive(ijk); }

	template <typename RealT, template <typename...> class Vec3T>
	__hostdev__ ValueT operator()(const Vec3T<RealT>& xyz) const {
		return BaseT::sample(xyz);
	}

	__hostdev__ ValueT operator()(const nanovdb::Coord& ijk) const {
		// Avoid unnecessary memory access if not active
		if (!isDataActive(ijk)) {
			return ValueT(0);
		}

		return BaseT::Acc()(ijk);
	}
};