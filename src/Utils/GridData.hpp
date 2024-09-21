//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <nanovdb/NanoVDB.h>
#include <openvdb/openvdb.h>

#pragma once

template <typename CoordT, typename ValueT>
struct GridData {
	CoordT* pCoords = nullptr;
	ValueT* pValues = nullptr;
	size_t size = 0;
};

using NanoFloatGrid = GridData<nanovdb::Coord, float>;
using NanoVectorGrid = GridData<nanovdb::Coord, nanovdb::Vec3f>;
using OpenFloatGrid = GridData<openvdb::Coord, float>;
using OpenVectorGrid = GridData<openvdb::Coord, openvdb::Vec3f>;
