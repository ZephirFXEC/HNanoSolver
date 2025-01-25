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

inline void extractFromGridIJK(const openvdb::VectorGrid::ConstPtr& domain, const openvdb::FloatGrid::ConstPtr& grid,
                               OpenGrid<float>& out_data) {
	const openvdb::VectorTree& domain_topology = domain->tree();
	const openvdb::FloatTree& grid_topology = grid->tree();

	out_data.allocateCudaPinned(domain_topology.activeVoxelCount());

	openvdb::tree::ValueAccessor<const openvdb::VectorTree> accessor_domain(domain_topology);
	const openvdb::tree::ValueAccessor<const openvdb::FloatTree> accessor_grid(grid_topology);

	openvdb::Coord* coords = out_data.pCoords();
	float* values = out_data.pValues();

	for (auto iter = domain_topology.cbeginValueOn(); iter; ++iter) {
		const openvdb::Coord coord = iter.getCoord();
		const size_t idx = (coord[0] & 7) << 6 | (coord[1] & 7) << 3 | coord[2] & 7;

		coords[idx] = coord;
		values[idx] = accessor_grid.getValue(coord);
	}
}


inline void extractToGlobalIdx(const openvdb::VectorGrid::ConstPtr& domain, const openvdb::FloatGrid::ConstPtr& grid,
                               IndexFloatGrid& out_data) {
	using DomainTree = openvdb::VectorGrid::TreeType;
	using DomainLeafNode = DomainTree::LeafNodeType;

	const DomainTree& domainTree = domain->tree();

	// LeafManager from the domain's tree
	const openvdb::tree::LeafManager<const DomainTree> leafManager(domainTree);
	const size_t numLeaves = leafManager.leafCount();

	std::vector<size_t> leafVoxelCounts(numLeaves);
	std::vector<size_t> leafOffsets(numLeaves + 1, 0);

	// find total active voxel count
	const size_t totalVoxels = tbb::parallel_reduce(
	    tbb::blocked_range<size_t>(0, numLeaves), static_cast<size_t>(0),
	    [&](const tbb::blocked_range<size_t>& r, size_t init) {
		    for (size_t i = r.begin(); i != r.end(); ++i) {
			    const DomainLeafNode& leaf = leafManager.leaf(i);
			    const size_t count = leaf.onVoxelCount();
			    leafVoxelCounts[i] = count;
			    init += count;
		    }
		    return init;
	    },
	    std::plus<>());
	out_data.size = totalVoxels;

	// compute leafOffsets
	tbb::parallel_scan(
	    tbb::blocked_range<size_t>(0, numLeaves), static_cast<size_t>(0),
	    [&](const tbb::blocked_range<size_t>& r, size_t sum, const bool isFinal) {
		    for (size_t i = r.begin(); i != r.end(); ++i) {
			    const size_t count = leafVoxelCounts[i];
			    if (isFinal) {
				    leafOffsets[i] = sum;
			    }
			    sum += count;
		    }
		    if (isFinal && r.end() == numLeaves) {
			    leafOffsets[numLeaves] = sum;
		    }
		    return sum;
	    },
	    std::plus<>());

	out_data.allocateCudaPinned(totalVoxels);

	// iterate each leaf in domain, gather voxel coords + values
	tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeaves), [&](const tbb::blocked_range<size_t>& range) {
		for (size_t i = range.begin(); i != range.end(); ++i) {
			// Retrieve this leaf node
			const DomainLeafNode& leaf = leafManager.leaf(i);

			// The base offset (prefix sum) for leaf i
			const size_t leafBaseOffset = leafOffsets[i];

			// We'll accumulate a local index from 0..(leafVoxelCounts[i]-1)
			size_t localIdx = 0;

			// Iterate over all active voxels in ascending coordinate order
			for (auto iter = leaf.cbeginValueOn(); iter.test(); ++iter) {
				// Compute the final "global" index for this voxel
				const size_t outIdx = leafBaseOffset + localIdx;

				// Store that global index in coords
				out_data.pCoords()[outIdx] = static_cast<uint32_t>(outIdx);

				// Retrieve the float value from 'grid' at the same coordinate
				const openvdb::Coord& c = iter.getCoord();
				const float val = grid->tree().getValue(c);

				// Write it to out_data.values
				out_data.pValues()[outIdx] = val;

				++localIdx;
			}
		}
	});
}


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
	    tbb::blocked_range<size_t>(0, numLeaves), static_cast<size_t>(0),
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
	    tbb::blocked_range<size_t>(0, numLeaves), static_cast<size_t>(0),
	    [&](const tbb::blocked_range<size_t>& range, size_t sum, const bool isFinalScan) {
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
	out_data.allocateCudaPinned(totalVoxels);

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