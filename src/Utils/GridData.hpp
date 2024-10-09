//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <nanovdb/NanoVDB.h>
#include <openvdb/openvdb.h>

#pragma once

template <typename CoordT, typename ValueT>
struct GridData {
	GridData() = default;
	GridData(const CoordT* pCoords, const ValueT* pValues, const size_t size)
	    : pCoords(pCoords), pValues(pValues), size(size) {}
	GridData(const GridData& other) : pCoords(other.pCoords), pValues(other.pValues), size(other.size) {}

	~GridData() {
		delete[] pCoords;
		delete[] pValues;
	}

	CoordT* pCoords = nullptr;
	ValueT* pValues = nullptr;
	size_t size = 0;
};

using NanoFloatGrid = GridData<nanovdb::Coord, float>;
using NanoVectorGrid = GridData<nanovdb::Coord, nanovdb::Vec3f>;
using OpenFloatGrid = GridData<openvdb::Coord, float>;
using OpenVectorGrid = GridData<openvdb::Coord, openvdb::Vec3f>;

template <size_t N, typename CoordT, typename ValueT>
struct MultiScalar {
	MultiScalar() = default;
	MultiScalar(const CoordT* pCoords, const ValueT* pValues, const size_t size) {
		for (size_t i = 0; i < N; ++i) {
			data[i] = GridData<CoordT, ValueT>(pCoords, pValues, size);
		}
	}

	~MultiScalar() = default;

	GridData<CoordT, ValueT>& operator[](size_t i) { return data[i]; }
	const GridData<CoordT, ValueT>& operator[](size_t i) const { return data[i]; }

private:
	std::array<GridData<CoordT, ValueT>, N> data;
};