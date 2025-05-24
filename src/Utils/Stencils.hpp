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
	using AccessorT = nanovdb::ReadAccessor<nanovdb::ValueOnIndex, 0, 1, 2>;


   public:
	__hostdev__ explicit IndexOffsetSampler(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* grid) : mAcc(*grid) {}

	__hostdev__ __forceinline__ uint64_t offset(const int i, const int j, const int k) const { return mAcc.getValue(i, j, k); }

	__hostdev__ __forceinline__ uint64_t offset(const nanovdb::Coord& ijk) const { return mAcc.getValue(ijk); }

	__hostdev__ __forceinline__ bool isActive(const int i, const int j, const int k) const {
		return mAcc.isActive(nanovdb::Coord(i, j, k));
	}

	__hostdev__ __forceinline__ bool isActive(const nanovdb::Coord& ijk) const { return mAcc.isActive(ijk); }

   private:
	AccessorT mAcc;
};


template <typename ValueT>
class IndexSampler<ValueT, 0> {
   public:
	__hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data)
	    : mOffsetSampler(offsetSampler), mData(data) {}


	__hostdev__ __forceinline__ ValueT operator()(const nanovdb::Coord& ijk) const {
		const auto off = mOffsetSampler.offset(ijk);
		return off == 0 ? ValueT(0) : mData[off - 1];
	}

	__hostdev__ __forceinline__ ValueT operator()(const int i, const int j, const int k) const {
		const auto off = mOffsetSampler.offset(i, j, k);
		return off == 0 ? ValueT(0) : mData[off - 1];
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
		// Gather values with direct coordinate access
		v[0][0][0] = mNearestSampler(ijk[0], ijk[1], ijk[2]);
		v[0][0][1] = mNearestSampler(ijk[0], ijk[1], ijk[2] + 1);
		v[0][1][0] = mNearestSampler(ijk[0], ijk[1] + 1, ijk[2]);
		v[0][1][1] = mNearestSampler(ijk[0], ijk[1] + 1, ijk[2] + 1);
		v[1][0][0] = mNearestSampler(ijk[0] + 1, ijk[1], ijk[2]);
		v[1][0][1] = mNearestSampler(ijk[0] + 1, ijk[1], ijk[2] + 1);
		v[1][1][0] = mNearestSampler(ijk[0] + 1, ijk[1] + 1, ijk[2]);
		v[1][1][1] = mNearestSampler(ijk[0] + 1, ijk[1] + 1, ijk[2] + 1);
	}

	// Changed sample from static to non-static so that it can call the stencil member function.
	template <typename RealT, template <typename...> class Vec3T>
	__hostdev__ ValueT sample(const Vec3T<RealT>& uvw) const {
		auto tmp = uvw;
		const nanovdb::Coord ijk = Floor(tmp);

		ValueT v[2][2][2];
		this->stencil(ijk, v);

		// Original generic lerp lambda (good for general types)
		// auto lerp_generic = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };

		// Lerp dispatch logic
		auto lerp_dispatch = [&](const ValueT& val_a, const ValueT& val_b, RealT weight) -> ValueT {
			if constexpr (std::is_same_v<ValueT, nanovdb::Vec3f> && std::is_same_v<RealT, float>) {
#if (defined(__CUDACC__) || defined(__HIP__)) && defined(__CUDA_ARCH__)
				// Assumes nanovdb::Vec3f has operator- defined that returns Vec3f
				nanovdb::Vec3f diff = val_b - val_a;  // val_b and val_a are const nanovdb::Vec3f&
				return ::fmaf(weight, diff, val_a);   // Calls your top-level fmaf
#else
				// Host or non-Vec3f version
				return val_a + (val_b - val_a) * weight;
#endif
			} else {
				return val_a + ValueT(weight) * (val_b - val_a);
			}
		};

		const ValueT z0 = lerp_dispatch(v[0][0][0], v[0][0][1], tmp[2]);
		const ValueT z1 = lerp_dispatch(v[0][1][0], v[0][1][1], tmp[2]);
		const ValueT z2 = lerp_dispatch(v[1][0][0], v[1][0][1], tmp[2]);
		const ValueT z3 = lerp_dispatch(v[1][1][0], v[1][1][1], tmp[2]);

		const ValueT y0 = lerp_dispatch(z0, z1, tmp[1]);
		const ValueT y1 = lerp_dispatch(z2, z3, tmp[1]);

		return lerp_dispatch(y0, y1, tmp[0]);
	}

   private:
	const IndexSampler<ValueT, 0> mNearestSampler;
};


template <typename ValueT>
class IndexSampler<ValueT, 1> : public TrilinearSampler<ValueT> {
	using BaseT = TrilinearSampler<ValueT>;

   public:
	__hostdev__ explicit IndexSampler(const IndexOffsetSampler<0>& offsetSampler, const ValueT* data) : BaseT(offsetSampler, data) {}

	template <typename RealT, template <typename...> class Vec3T>
	__hostdev__ ValueT operator()(const Vec3T<RealT>& xyz) const {
		return BaseT::sample(xyz);
	}

	__hostdev__ ValueT operator()(const nanovdb::Coord& ijk) const { return BaseT::Acc()(ijk); }
};