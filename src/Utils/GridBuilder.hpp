//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <openvdb/openvdb.h>
#include <tbb/tbb.h>

#include <utility>

#include "GridData.hpp"

namespace HNS {

template <typename DomainGridT>
class IndexGridBuilder {
   public:
	explicit IndexGridBuilder(typename DomainGridT::ConstPtr domain, GridIndexedData* data)
	    : m_domainGrid(std::move(domain)), m_outData(data) {
		if (!m_domainGrid) {
			throw std::runtime_error("IndexGridBuilder: domain grid is null!");
		}

		computeLeafs();
		m_outData->allocateCoords(m_totalVoxels, AllocationType::CudaPinned);
	}

	~IndexGridBuilder() {
		free(m_leafVoxelCounts);
		free(m_leafOffsets);
	}

	void addGrid(openvdb::GridBase::Ptr grid, const std::string& name) {
		if (!grid) {
			throw std::runtime_error("IndexGridBuilder: Grid is null ! Couldn't add grid");
		}
		m_grids.emplace_back(std::move(grid), name);
	}

	void build() {
		for (const auto& [grid, name] : m_grids) {
			if (const auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
				m_outData->addValueBlock<float>(name, AllocationType::CudaPinned, m_totalVoxels);
				auto& floatTree = floatGrid->tree();
				float* outPtr = m_outData->pValues<float>(name);
				m_samplers.emplace_back(
				    [outPtr, &floatTree](const openvdb::Coord& c, const size_t idx) { outPtr[idx] = floatTree.getValue(c); });
			} else if (const auto vecGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
				m_outData->addValueBlock<openvdb::Vec3f>(name, AllocationType::CudaPinned, m_totalVoxels);
				auto& vecTree = vecGrid->tree();
				openvdb::Vec3f* outPtr = m_outData->pValues<openvdb::Vec3f>(name);
				m_samplers.emplace_back(
				    [outPtr, &vecTree](const openvdb::Coord& c, const size_t idx) { outPtr[idx] = vecTree.getValue(c); });
			} else {
				throw std::runtime_error("IndexGridBuilder: unsupported grid type!");
			}
		}

		using TreeType = typename DomainGridT::TreeType;
		const TreeType& tree = m_domainGrid->tree();
		const openvdb::tree::LeafManager<const TreeType> leafManager(tree);
		const size_t numLeaves = leafManager.leafCount();

		parallel_for(tbb::blocked_range<size_t>(0, numLeaves), [&](const tbb::blocked_range<size_t>& range) {
			for (size_t i = range.begin(); i != range.end(); ++i) {
				const auto& leaf = leafManager.leaf(i);
				const size_t leafBaseOffset = m_leafOffsets[i];
				size_t localIdx = 0;

				for (auto iter = leaf.cbeginValueOn(); iter.test(); ++iter) {
					const size_t outIdx = leafBaseOffset + localIdx;

					m_outData->pIndexes()[outIdx] = static_cast<uint32_t>(outIdx);

					const openvdb::Coord& c = iter.getCoord();

					m_outData->pCoords()[outIdx] = c;

					for (const auto& sampler : m_samplers) {
						sampler(c, outIdx);
					}

					++localIdx;
				}
			}
		});
	}


	template <typename GridT, typename ValueT = typename GridT::ValueType>
	typename GridT::Ptr writeIndexGrid(const std::string& name, const float voxelSize) {
		typename GridT::Ptr grid = GridT::create();
		grid->setName(name);
		grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));

		if constexpr (std::is_same_v<GridT, openvdb::FloatGrid>) {
			grid->setGridClass(openvdb::GRID_FOG_VOLUME);
		} else if constexpr (std::is_same_v<GridT, openvdb::VectorGrid>) {
			grid->setGridClass(openvdb::GRID_STAGGERED);
			grid->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE);
		}

		auto& newTree = grid->tree();
		const auto& domainTree = m_domainGrid->tree();
		openvdb::tree::ValueAccessor<typename GridT::TreeType> accessor(newTree);
		const openvdb::tree::LeafManager<const typename DomainGridT::TreeType> domainLeafManager(domainTree);

		if (m_totalVoxels != m_outData->size()) {
			throw std::runtime_error("Mismatch between domain grid active voxel count and index grid data size");
		}

		const auto* data = m_outData->pValues<ValueT>(name);

		for (size_t leafIdx = 0; leafIdx < m_numLeaves; ++leafIdx) {
			const auto& leaf = domainLeafManager.leaf(leafIdx);
			const size_t leafBaseOffset = m_leafOffsets[leafIdx];
			size_t localIndex = 0;

			for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
				const size_t globalIndex = leafBaseOffset + localIndex;
				const openvdb::Coord& coord = iter.getCoord();

				accessor.setValue(coord, data[globalIndex]);
				++localIndex;
			}
		}

		return grid;
	}


   private:
	void computeLeafs() {
		using TreeType = typename DomainGridT::TreeType;
		const TreeType& tree = m_domainGrid->tree();
		const openvdb::tree::LeafManager<const TreeType> leafManager(tree);
		m_numLeaves = leafManager.leafCount();

		m_leafVoxelCounts = static_cast<size_t*>(malloc(m_numLeaves * sizeof(size_t)));
		m_leafOffsets = static_cast<size_t*>(malloc((m_numLeaves + 1) * sizeof(size_t)));

		m_totalVoxels = parallel_reduce(
		    tbb::blocked_range<size_t>(0, m_numLeaves), static_cast<size_t>(0),
		    [&](const tbb::blocked_range<size_t>& r, size_t init) {
			    for (size_t i = r.begin(); i != r.end(); ++i) {
				    const auto& leaf = leafManager.leaf(i);
				    const size_t count = leaf.onVoxelCount();
				    m_leafVoxelCounts[i] = count;
				    init += count;
			    }
			    return init;
		    },
		    std::plus<>());

		parallel_scan(
		    tbb::blocked_range<size_t>(0, m_numLeaves), static_cast<size_t>(0),
		    [&](const tbb::blocked_range<size_t>& r, size_t sum, const bool isFinal) {
			    for (size_t i = r.begin(); i != r.end(); ++i) {
				    const size_t count = m_leafVoxelCounts[i];
				    if (isFinal) {
					    m_leafOffsets[i] = sum;
				    }
				    sum += count;
			    }
			    if (isFinal && r.end() == m_numLeaves) {
				    m_leafOffsets[m_numLeaves] = sum;
			    }
			    return sum;
		    },
		    std::plus<>());
	}

	typename DomainGridT::ConstPtr m_domainGrid;
	size_t* m_leafVoxelCounts = nullptr;
	size_t* m_leafOffsets = nullptr;
	size_t m_totalVoxels = 0;
	size_t m_numLeaves = 0;
	GridIndexedData* m_outData;

	std::vector<std::function<void(const openvdb::Coord&, size_t)>> m_samplers;
	std::vector<std::pair<openvdb::GridBase::Ptr, std::string>> m_grids;
};

inline void extractToGlobalIdx(const openvdb::VectorGrid::ConstPtr& domain, const openvdb::FloatGrid::ConstPtr& grid,
                               GridIndexedData& out_data) {
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

	out_data.addValueBlock<float>(grid->getName(), AllocationType::CudaPinned, totalVoxels);
	out_data.addValueBlock<openvdb::Vec3f>(domain->getName(), AllocationType::CudaPinned, totalVoxels);
	out_data.allocateCoords(totalVoxels, AllocationType::CudaPinned);

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
				out_data.pIndexes()[outIdx] = outIdx;

				// Retrieve the float value from 'grid' at the same coordinate
				const openvdb::Coord& c = iter.getCoord();
				const float val = grid->tree().getValue(c);
				out_data.pCoords()[outIdx] = c;

				const openvdb::Vec3f vec = domain->tree().getValue(c);

				// Write it to out_data.values
				out_data.pValues<float>(grid->getName())[outIdx] = val;
				out_data.pValues<openvdb::Vec3f>(domain->getName())[outIdx] = vec;

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