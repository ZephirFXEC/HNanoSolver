//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tree/NodeManager.h>
#include <concurrent_vector.h>

#include <vector>
#include <memory>

// iterate over openvdb grid
// launch a kernel with the [pos, value] pairs
// use cudaVoxelToGrid to write to nanovdb grid
// ... calculations
// write back to a [pos, value] to build openvdb grid

// see : https://github.com/danrbailey/siggraph2023_openvdb/blob/master/benchmarks/cloud_value_clamp/main.cpp
// for Multi-threaded OpenVDB iteration

inline void extractFromOpenVDB(const openvdb::FloatGrid::ConstPtr& grid, std::vector<openvdb::Coord>& pos, std::vector<float>& value) {


	const openvdb::FloatTree& tree = grid->tree();

	// Estimate the size to avoid frequent reallocations
	const size_t estimatedSize = tree.activeVoxelCount();

	// Create a memory resource (e.g., a monotonic buffer)
	std::pmr::monotonic_buffer_resource mbr;

	// Create pmr::vectors using the custom memory resource
	std::pmr::vector<openvdb::Coord> posVector(&mbr);
	std::pmr::vector<float> valueVector(&mbr);

	posVector.reserve(estimatedSize);
	valueVector.reserve(estimatedSize);

	auto getOp = [&](const auto& node) {
		for (auto iter = node.beginValueOn(); iter; ++iter) {
			posVector.push_back(iter.getCoord());
			valueVector.push_back(iter.getValue());
		}
	};

	openvdb::tree::NodeManager<const openvdb::FloatTree> nodeManager(tree);
	nodeManager.foreachTopDown(getOp);

	pos = std::vector<openvdb::Coord>(posVector.begin(), posVector.end());
	value = std::vector<float>(valueVector.begin(), valueVector.end());
}
