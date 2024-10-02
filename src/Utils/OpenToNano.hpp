//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <concurrent_vector.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/NodeManager.h>

#include <execution>

#include "GridData.hpp"

// iterate over openvdb grid
// launch a kernel with the [pos, value] pairs
// use cudaVoxelToGrid to write to nanovdb grid
// ... calculations
// write back to a [pos, value] to build openvdb grid

// see : https://github.com/danrbailey/siggraph2023_openvdb/blob/master/benchmarks/cloud_value_clamp/main.cpp
// for Multi-threaded OpenVDB iteration

template <typename GridT, typename CoordT, typename ValueT>
inline void extractFromOpenVDB(const typename GridT::ConstPtr& grid, GridData<CoordT, ValueT>& out_data) {
	const auto& tree = grid->tree();
	out_data.size = tree.activeVoxelCount();
	out_data.pCoords = new CoordT[out_data.size];
	out_data.pValues = new ValueT[out_data.size];

	std::atomic<size_t> writePos{0};
	constexpr size_t chunkSize = 64;

	openvdb::tree::NodeManager<const typename GridT::TreeType> nodeManager(tree);
	nodeManager.foreachTopDown([&](const auto& node) {
		std::vector<CoordT> localCoords;
		std::vector<ValueT> localValues;
		localCoords.reserve(chunkSize);  // Avoid dynamic resizing
		localValues.reserve(chunkSize);

		for (auto iter = node.cbeginValueOn(); iter; ++iter) {
			localCoords.push_back(iter.getCoord());
			localValues.push_back(iter.getValue());

			if (localCoords.size() >= chunkSize) {
				const size_t pos = writePos.fetch_add(chunkSize);
				std::copy(localCoords.begin(), localCoords.end(), out_data.pCoords + pos);
				std::copy(localValues.begin(), localValues.end(), out_data.pValues + pos);
				localCoords.clear();
				localValues.clear();
			}
		}

		if (!localCoords.empty()) {
			const size_t pos = writePos.fetch_add(localCoords.size());
			std::copy(localCoords.begin(), localCoords.end(), out_data.pCoords + pos);
			std::copy(localValues.begin(), localValues.end(), out_data.pValues + pos);
		}
	});
}

