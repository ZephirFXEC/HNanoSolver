//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tree/NodeManager.h>

// iterate over openvdb grid
// launch a kernel with the [pos, value] pairs
// use cudaVoxelToGrid to write to nanovdb grid
// ... calculations
// write back to a [pos, value] to build openvdb grid

// see : https://github.com/danrbailey/siggraph2023_openvdb/blob/master/benchmarks/cloud_value_clamp/main.cpp
// for Multi-threaded OpenVDB iteration

inline void extractFromOpenVDB(const openvdb::FloatGrid::ConstPtr& grid, openvdb::Coord* pos, float* value,
                               size_t size) {


	openvdb::FloatTree treeCopy(grid->tree());

	auto getOp = [&](auto& node) {
		for (auto iter = node.beginValueOn(); iter; ++iter) {

		}
	};


	openvdb::tree::NodeManager<openvdb::FloatTree> nodeManager(treeCopy);
	nodeManager.foreachTopDown(getOp);
}

