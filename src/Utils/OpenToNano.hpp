//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <openvdb/openvdb.h>
#include <tbb/tbb.h>

#include "GridData.hpp"

// iterate over openvdb grid
// launch a kernel with the [pos, value] pairs
// use cudaVoxelToGrid to write to nanovdb grid
// ... calculations
// write back to a [pos, value] to build openvdb grid

// see : https://github.com/danrbailey/siggraph2023_openvdb/blob/master/benchmarks/cloud_value_clamp/main.cpp
// for Multi-threaded OpenVDB iteration

namespace HNS {
template <typename GridT, typename ValueT>
void extractFromOpenVDB(const typename GridT::ConstPtr& grid, OpenGrid<ValueT>& out_data) {
	using TreeType = typename GridT::TreeType;
	using LeafNodeType = typename TreeType::LeafNodeType;
	const TreeType& tree = grid->tree();

	// Create a LeafManager to efficiently manage leaf nodes
	openvdb::tree::LeafManager<const TreeType> leafManager(tree);

	const size_t numLeaves = leafManager.leafCount();

	// Vectors to hold per-leaf voxel counts and offsets
	std::vector<size_t> leafVoxelCounts(numLeaves);
	std::vector<size_t> leafOffsets(numLeaves + 1, 0);  // +1 for total voxels

	// Compute per-leaf voxel counts and total voxel count in parallel
	size_t totalVoxels = tbb::parallel_reduce(
	    tbb::blocked_range<size_t>(0, numLeaves), size_t(0),
	    [&](const tbb::blocked_range<size_t>& range, size_t init) {
		    for (size_t i = range.begin(); i != range.end(); ++i) {
			    const LeafNodeType& leaf = leafManager.leaf(i);
			    const size_t count = leaf.onVoxelCount();
			    leafVoxelCounts[i] = count;
			    init += count;
		    }
		    return init;
	    },
	    std::plus<size_t>());

	// Compute prefix sums (leafOffsets) in parallel
	tbb::parallel_scan(
	    tbb::blocked_range<size_t>(0, numLeaves), size_t(0),
	    [&](const tbb::blocked_range<size_t>& range, size_t sum, bool isFinalScan) {
		    for (size_t i = range.begin(); i != range.end(); ++i) {
			    const size_t count = leafVoxelCounts[i];
			    if (isFinalScan) {
				    leafOffsets[i] = sum;
			    }
			    sum += count;
		    }
		    if (isFinalScan && range.end() == numLeaves) {
			    leafOffsets[numLeaves] = sum;  // Total voxels
		    }
		    return sum;
	    },
	    std::plus<size_t>());

	// Allocate output arrays with pinned memory
	out_data.allocateStandard(totalVoxels);

	// Process leaves in parallel to extract voxel data
	tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeaves), [&](const tbb::blocked_range<size_t>& range) {
		for (size_t i = range.begin(); i != range.end(); ++i) {
			const LeafNodeType& leaf = leafManager.leaf(i);
			size_t offset = leafOffsets[i];

			openvdb::Coord* coords = out_data.pCoords() + offset;
			ValueT* values = out_data.pValues() + offset;

			size_t idx = 0;
			for (auto iter = leaf.cbeginValueOn(); iter.test(); ++iter) {
				coords[idx] = iter.getCoord();
				values[idx] = iter.getValue();
				++idx;
			}
		}
	});
}


}  // namespace HNS