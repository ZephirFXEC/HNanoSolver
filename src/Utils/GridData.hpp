//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <nanovdb/NanoVDB.h>
#include <openvdb/openvdb.h>
namespace HNS {
static constexpr size_t ALIGNMENT = 32;
template <typename CoordT, typename ValueT>
struct GridData {
	GridData() = default;
	GridData(const CoordT* pCoords, const ValueT* pValues, const size_t size)
	    : pCoords(pCoords), pValues(pValues), size(size) {}
	GridData(const GridData& other) : pCoords(other.pCoords), pValues(other.pValues), size(other.size) {}

	void* operator new(const size_t size) {
		void* ptr = nullptr;
		ptr = _aligned_malloc(size, ALIGNMENT);
		if (!ptr) throw std::bad_alloc();
		return ptr;
	}
	void* operator new[](const size_t size) {
		void* ptr = nullptr;
		ptr = _aligned_malloc(size, ALIGNMENT);
		if (!ptr) throw std::bad_alloc();
		return ptr;
	}

	void operator delete(void* ptr) { _aligned_free(ptr); }
	void operator delete[](void* ptr) { _aligned_free(ptr); }

	~GridData() {
		delete[] pCoords;
		delete[] pValues;
		size = 0;
	}

	CoordT* pCoords = nullptr;
	ValueT* pValues = nullptr;
	size_t size = 0;
};

// Base templates for OpenVDB and NanoVDB grids
template <typename CoordType, typename ValueType>
using GenericGrid = GridData<CoordType, ValueType>;

template <typename T>
using OpenGrid = GenericGrid<openvdb::Coord, T>;

template <typename T>
using NanoGrid = GenericGrid<nanovdb::Coord, T>;

// Specific grid types with float and vector values
template <typename CoordType>
using FloatGrid = GenericGrid<CoordType, float>;

template <typename CoordType>
using VectorGrid = GenericGrid<CoordType, nanovdb::Vec3f>;

// Typedefs for common OpenVDB and NanoVDB grids
using OpenFloatGrid = OpenGrid<float>;
using OpenVectorGrid = OpenGrid<openvdb::Vec3f>;

using NanoFloatGrid = NanoGrid<float>;
using NanoVectorGrid = NanoGrid<nanovdb::Vec3f>;


}  // namespace HNS