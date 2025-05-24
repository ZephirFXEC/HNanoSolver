//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/Activate.h>
#include <openvdb/tools/Prune.h>
#include <tbb/tbb.h>

#include <execution>
#include <utility>

#include "GridData.hpp"
#include "ScopedTimer.hpp"

namespace HNS {

template <typename DomainGridT>
class IndexGridBuilder {
   public:
	using DomainGridPtr = typename DomainGridT::ConstPtr;
	using TreeType = typename DomainGridT::TreeType;
	using LeafNodeType = typename TreeType::LeafNodeType;

	// Grid variant for type-safe storage
	using GridVariant = std::variant<openvdb::FloatGrid::Ptr, openvdb::VectorGrid::Ptr>;

	explicit IndexGridBuilder(DomainGridPtr domain, GridIndexedData* data) : m_domainGrid{std::move(domain)}, m_outData{data} {
		if (!m_domainGrid) {
			throw std::runtime_error("IndexGridBuilder: domain grid is null!");
		}

		if (!m_outData) {
			throw std::runtime_error("IndexGridBuilder: output data is null!");
		}

		computeLeafs();
		m_outData->allocateCoords(m_totalVoxels);
	}

	IndexGridBuilder() = default;

	// Non-copyable but movable
	IndexGridBuilder(const IndexGridBuilder&) = delete;
	IndexGridBuilder& operator=(const IndexGridBuilder&) = delete;
	IndexGridBuilder(IndexGridBuilder&&) = default;
	IndexGridBuilder& operator=(IndexGridBuilder&&) = default;


	void addGrid(const openvdb::GridBase::Ptr& grid, std::string_view name) {
		const ScopedTimer timer{std::string{"IndexGridBuilder::AddGrid "} + std::string{name}};

		if (!grid) {
			throw std::runtime_error("IndexGridBuilder: Grid is null! Couldn't add grid");
		}

		// Use structured bindings and if constexpr for cleaner type handling
		if (auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
			addGridFloat(std::move(floatGrid), name);
		} else if (auto vectorGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
			addGridVector(std::move(vectorGrid), name);
		} else {
			throw std::runtime_error("IndexGridBuilder: Unsupported grid type");
		}
	}

	void addGridSDF(openvdb::FloatGrid::Ptr grid, std::string_view name) {
		const std::string nameStr{name};
		m_sdf_grids.emplace(nameStr, std::move(grid));
		m_outData->addValueBlock<float>(nameStr, m_totalVoxels);
	}

	void addGridFloat(openvdb::FloatGrid::Ptr grid, std::string_view name) {
		const std::string nameStr{name};
		m_float_grids.emplace(nameStr, std::move(grid));
		m_outData->addValueBlock<float>(nameStr, m_totalVoxels);
	}

	void addGridVector(openvdb::VectorGrid::Ptr grid, std::string_view name) {
		const std::string nameStr{name};
		m_vector_grids.emplace(nameStr, std::move(grid));
		m_outData->addValueBlock<openvdb::Vec3f>(nameStr, m_totalVoxels);
	}

	void build() {
		ScopedTimer timer("IndexGridBuilder::Build");

		using TreeType = typename DomainGridT::TreeType;
		const TreeType& tree = m_domainGrid->tree();
		const openvdb::tree::LeafManager<const TreeType> leafManager(tree);

		for (const auto& [name, grid] : m_sdf_grids) {
			const auto& floatTree = grid->tree();
			const openvdb::tree::LeafManager floatLeafManager(floatTree);
			float* outPtr = m_outData->pValues<float>(name);

			parallel_for(tbb::blocked_range<size_t>(0, m_numLeaves), [&](const tbb::blocked_range<size_t>& range) {
				for (size_t i = range.begin(); i != range.end(); ++i) {
					const auto& leaf = leafManager.leaf(i);
					const size_t leafBaseOffset = m_leafOffsets[i];
					const openvdb::Coord leafPos = leaf.origin();

					if (const auto grid_leaf = floatTree.probeConstLeaf(leafPos); grid_leaf != nullptr) {
						std::memcpy(outPtr + leafBaseOffset, grid_leaf->buffer().data(), TreeType::LeafNodeType::SIZE * sizeof(float));
					} else {
						std::memset(outPtr + leafBaseOffset, 1, TreeType::LeafNodeType::SIZE * sizeof(float));
					}
				}
			});
		}

		for (const auto& [name, grid] : m_float_grids) {
			const auto& floatTree = grid->tree();
			const openvdb::tree::LeafManager floatLeafManager(floatTree);
			float* outPtr = m_outData->pValues<float>(name);

			parallel_for(tbb::blocked_range<size_t>(0, m_numLeaves), [&](const tbb::blocked_range<size_t>& range) {
				for (size_t i = range.begin(); i != range.end(); ++i) {
					const auto& leaf = leafManager.leaf(i);
					const size_t leafBaseOffset = m_leafOffsets[i];
					const openvdb::Coord leafPos = leaf.origin();

					if (const auto grid_leaf = floatTree.probeConstLeaf(leafPos); grid_leaf != nullptr) {
						std::memcpy(outPtr + leafBaseOffset, grid_leaf->buffer().data(), TreeType::LeafNodeType::SIZE * sizeof(float));
					} else {
						std::memset(outPtr + leafBaseOffset, 0, TreeType::LeafNodeType::SIZE * sizeof(float));
					}
				}
			});
		}


		for (const auto& [name, grid] : m_vector_grids) {
			const auto& vecTree = grid->tree();
			const openvdb::tree::LeafManager floatLeafManager(vecTree);
			openvdb::Vec3f* outPtr = m_outData->pValues<openvdb::Vec3f>(name);

			parallel_for(tbb::blocked_range<size_t>(0, m_numLeaves), [&](const tbb::blocked_range<size_t>& range) {
				for (size_t i = range.begin(); i != range.end(); ++i) {
					const auto& leaf = leafManager.leaf(i);
					const size_t leafBaseOffset = m_leafOffsets[i];
					const openvdb::Coord leafPos = leaf.origin();

					if (const auto grid_leaf = vecTree.probeConstLeaf(leafPos); grid_leaf != nullptr) {
						std::memcpy(outPtr + leafBaseOffset, grid_leaf->buffer().data(),
						            TreeType::LeafNodeType::SIZE * sizeof(openvdb::Vec3f));
					} else {
						std::memset(outPtr + leafBaseOffset, 0, TreeType::LeafNodeType::SIZE * sizeof(openvdb::Vec3f));
					}
				}
			});
		}

		parallel_for(tbb::blocked_range<size_t>(0, m_numLeaves), [&](const tbb::blocked_range<size_t>& range) {
			for (size_t i = range.begin(); i != range.end(); ++i) {
				const auto& leaf = leafManager.leaf(i);
				const size_t leafBaseOffset = m_leafOffsets[i];
				for (size_t j = 0; j < TreeType::LeafNodeType::SIZE; ++j) {
					const openvdb::Coord c = leaf.offsetToGlobalCoord(j);
					const size_t outIdx = leafBaseOffset + j;
					m_outData->pCoords()[outIdx] = c;
				}
			}
		});
	}

	template <typename GridT>
	typename GridT::Ptr writeIndexGrid(const std::string& name, const float voxelSize) {
		using ValueT = typename GridT::ValueType;
		using TreeT = typename GridT::TreeType;
		using LeafT = typename TreeT::LeafNodeType;

		ScopedTimer timer("IndexGridBuilder::WriteIndexGrid " + name);

		typename GridT::Ptr grid = GridT::create();

		grid->setName(name);
		grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));

		if constexpr (std::is_same_v<GridT, openvdb::FloatGrid>) {
			grid->setGridClass(openvdb::GRID_FOG_VOLUME);
		} else if constexpr (std::is_same_v<GridT, openvdb::VectorGrid>) {
			grid->setGridClass(openvdb::GRID_STAGGERED);
			grid->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE);
		}

		if (m_totalVoxels != m_outData->size()) {
			throw std::runtime_error("Mismatch between domain grid active voxel count and index grid data size");
		}

		auto& newTree = grid->tree();
		const auto& domainTree = m_domainGrid->tree();
		const auto* data = m_outData->pValues<ValueT>(name);

		// 1. Clone leaf structure from domain grid
		newTree.topologyUnion(domainTree);

		// 2. Use LeafManager for parallel data population
		openvdb::tree::LeafManager<TreeT> leafManager(newTree);
		openvdb::tree::LeafManager<const typename DomainGridT::TreeType> domainLeafManager(domainTree);

		// Process leaves in parallel
		leafManager.foreach (
		    [&data, this](LeafT& leaf, const size_t leafIdx) {
			    const size_t leafBaseOffset = m_leafOffsets[leafIdx];
			    ValueT* leafData = leaf.buffer().data();
			    std::memcpy(leafData, data + leafBaseOffset, LeafT::SIZE * sizeof(ValueT));
		    },
		    true, 12);

		// openvdb::tools::deactivate(grid->tree(), ValueT(0), ValueT(0), true);
		// openvdb::tools::pruneInactive(grid->tree());
		return grid;
	}

	void setAllocType(const AllocationType type) const { m_outData->setAllocationType(type); }

   private:
	void computeLeafs() {
		const ScopedTimer timer{"IndexGridBuilder::computeLeafs"};

		const auto& tree = m_domainGrid->tree();
		const openvdb::tree::LeafManager leafManager{tree};

		m_numLeaves = leafManager.leafCount();
		m_leafOffsets.resize(m_numLeaves + 1);
		m_totalVoxels = m_numLeaves * LeafNodeType::SIZE;

		// Use iota and transform for cleaner offset calculation
		std::vector<size_t> indices(m_numLeaves);
		std::iota(indices.begin(), indices.end(), 0);

		std::transform(std::execution::par_unseq, indices.begin(), indices.end(), m_leafOffsets.begin(),
		               [](size_t i) { return i * LeafNodeType::SIZE; });

		m_leafOffsets[m_numLeaves] = m_totalVoxels;
	}

	const typename DomainGridT::ConstPtr m_domainGrid;
	std::vector<size_t> m_leafOffsets{};
	size_t m_totalVoxels = 0;
	size_t m_numLeaves = 0;
	GridIndexedData* const m_outData;

	std::unordered_map<std::string, openvdb::FloatGrid::Ptr> m_sdf_grids;
	std::unordered_map<std::string, openvdb::FloatGrid::Ptr> m_float_grids;
	std::unordered_map<std::string, openvdb::VectorGrid::Ptr> m_vector_grids;
};


}  // namespace HNS