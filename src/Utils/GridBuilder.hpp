//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/Activate.h>
#include <openvdb/tools/Prune.h>
#include <tbb/tbb.h>

#include <utility>

#include "GridData.hpp"
#include "ScopedTimer.hpp"

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
		m_outData->allocateCoords(m_totalVoxels);
	}

	~IndexGridBuilder() = default;

	void addGrid(const openvdb::GridBase::Ptr& grid, const std::string& name) {
		ScopedTimer timer("IndexGridBuilder::AddGrid " + name);

		if (!grid) {
			throw std::runtime_error("IndexGridBuilder: Grid is null ! Couldn't add grid");
		}

		if (const auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
			addGridFloat(floatGrid, name);
		} else if (const auto vectorGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
			addGridVector(vectorGrid, name);
		} else {
			throw std::runtime_error("IndexGridBuilder: Unsupported grid type");
		}
	}

	void addGridFloat(openvdb::FloatGrid::Ptr grid, const std::string& name) {
		m_float_grids.insert({name, std::move(grid)});
		m_outData->addValueBlock<float>(name, m_totalVoxels);
	}

	void addGridVector(openvdb::VectorGrid::Ptr grid, const std::string& name) {
		m_vector_grids.insert({name, std::move(grid)});
		m_outData->addValueBlock<openvdb::Vec3f>(name, m_totalVoxels);
	}

	void build() {
		ScopedTimer timer("IndexGridBuilder::Build");

		for (const auto& [name, grid] : m_float_grids) {
			auto& floatTree = grid->tree();
			float* outPtr = m_outData->pValues<float>(name);
			m_samplers.emplace_back(
			    [outPtr, &floatTree](const openvdb::Coord& c, const size_t idx) { outPtr[idx] = floatTree.getValue(c); });
		}

		for (const auto& [name, grid] : m_vector_grids) {
			auto& vecTree = grid->tree();
			openvdb::Vec3f* outPtr = m_outData->pValues<openvdb::Vec3f>(name);
			m_samplers.emplace_back([outPtr, &vecTree](const openvdb::Coord& c, const size_t idx) { outPtr[idx] = vecTree.getValue(c); });
		}

		using TreeType = typename DomainGridT::TreeType;
		const TreeType& tree = m_domainGrid->tree();
		const openvdb::tree::LeafManager<const TreeType> leafManager(tree);

		parallel_for(tbb::blocked_range<size_t>(0, m_numLeaves), [&](const tbb::blocked_range<size_t>& range) {
			for (size_t i = range.begin(); i != range.end(); ++i) {
				const auto& leaf = leafManager.leaf(i);
				const size_t leafBaseOffset = m_leafOffsets[i];
				size_t localIdx = 0;
				
				for (auto iter = leaf.cbeginValueOn(); iter.test(); ++iter) {
					const size_t outIdx = leafBaseOffset + localIdx;

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
		leafManager.foreach ([&data, this](LeafT& leaf, const size_t leafIdx) {
			const size_t leafBaseOffset = m_leafOffsets[leafIdx];
			const auto& valueMask = leaf.getValueMask();
			ValueT* leafData = leaf.buffer().data();

			if (valueMask.isOn()) {
				std::memcpy(leafData, data + leafBaseOffset, LeafT::SIZE * sizeof(ValueT));
			} else {
				size_t localIndex = 0;
				for (auto iter = valueMask.beginOn(); iter; ++iter) {
					leafData[iter.pos()] = data[leafBaseOffset + localIndex];
					++localIndex;
				}
			}
		});

		return grid;
	}

	void setAllocType(const AllocationType type) const { m_outData->setAllocationType(type); }

   private:
	void computeLeafs() {
		ScopedTimer timer("IndexGridBuilder::computeLeafs");
		using TreeType = typename DomainGridT::TreeType;
		const TreeType& tree = m_domainGrid->tree();
		const openvdb::tree::LeafManager<const TreeType> leafManager(tree);

		leafManager.getPrefixSum(m_leafOffsets, m_numLeaves);

		m_totalVoxels = m_domainGrid->activeVoxelCount();
	}

	typename DomainGridT::ConstPtr m_domainGrid;
	size_t* m_leafOffsets{};
	size_t m_totalVoxels = 0;
	size_t m_numLeaves = 0;
	GridIndexedData* m_outData;

	std::unordered_map<std::string, openvdb::FloatGrid::Ptr> m_float_grids;
	std::unordered_map<std::string, openvdb::VectorGrid::Ptr> m_vector_grids;
	std::vector<std::function<void(const openvdb::Coord&, size_t)>> m_samplers;
};


/*
template <typename DomainGridT>
class IndexGridBuilder {
   public:
    struct BufferInfo {
        void* destPtr;
        size_t elementSize;
        const void* sourceBuffer;
    };

    explicit IndexGridBuilder(typename DomainGridT::ConstPtr domain, GridIndexedData* data)
        : m_domainGrid(std::move(domain)), m_outData(data) {
        if (!m_domainGrid) {
            throw std::runtime_error("IndexGridBuilder: domain grid is null!");
        }

        computeLeafs();
        m_outData->allocateCoords(m_totalVoxels);
    }

    ~IndexGridBuilder() = default;


    void addGrid(const openvdb::GridBase::Ptr& grid, const std::string& name) {
        ScopedTimer timer("IndexGridBuilder::AddGrid " + name);

        if (!grid) {
            throw std::runtime_error("IndexGridBuilder: Grid is null ! Couldn't add grid");
        }

        if (const auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
            addGridFloat(floatGrid, name);
        } else if (const auto vectorGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
            addGridVector(vectorGrid, name);
        } else {
            throw std::runtime_error("IndexGridBuilder: Unsupported grid type");
        }
    }

    void addGridFloat(openvdb::FloatGrid::Ptr grid, const std::string& name) {
        m_float_grids.insert({name, std::move(grid)});
        m_outData->addValueBlock<float>(name, m_totalVoxels);
    }

    void addGridVector(openvdb::VectorGrid::Ptr grid, const std::string& name) {
        m_vector_grids.insert({name, std::move(grid)});
        m_outData->addValueBlock<openvdb::Vec3f>(name, m_totalVoxels);
    }

    void buildVector(std::vector<std::vector<BufferInfo>>& leafBuffers,
                     const openvdb::tree::LeafManager<const typename DomainGridT::TreeType>& leafManager) {
        ScopedTimer timer("IndexGridBuilder::build::Vector");
        for (const auto& [name, grid] : m_vector_grids) {
            auto& vecTree = grid->tree();
            openvdb::Vec3f* outPtr = m_outData->pValues<openvdb::Vec3f>(name);

            for (size_t i = 0; i < m_numLeaves; ++i) {
                const auto& leaf = leafManager.leaf(i);
                if (const auto* vecLeaf = vecTree.probeConstLeaf(leaf.origin())) {
                    leafBuffers[i].push_back({outPtr + m_leafOffsets[i], sizeof(openvdb::Vec3f), vecLeaf->buffer().data()});
                }
            }
        }
    }

    void buildFloat(std::vector<std::vector<BufferInfo>>& leafBuffers,
                    const openvdb::tree::LeafManager<const typename DomainGridT::TreeType>& leafManager) {
        ScopedTimer timer("IndexGridBuilder::build::Float");
        for (const auto& [name, grid] : m_float_grids) {
            auto& floatTree = grid->tree();
            float* outPtr = m_outData->pValues<float>(name);
            openvdb::tree::LeafManager<const openvdb::FloatGrid::TreeType> readManager(floatTree);

            // For each leaf, store the buffer information
            for (size_t i = 0; i < m_numLeaves; ++i) {
                const auto& leaf = leafManager.leaf(i);
                if (const auto* floatLeaf = floatTree.probeConstLeaf(leaf.origin())) {
                    leafBuffers[i].push_back({outPtr + m_leafOffsets[i], sizeof(float), floatLeaf->buffer().data()});
                }
            }
        }
    }


    void build() {
        using TreeType = typename DomainGridT::TreeType;
        const TreeType& tree = m_domainGrid->tree();
        const openvdb::tree::LeafManager<const TreeType> leafManager(tree);

        std::vector<std::vector<BufferInfo>> leafBuffers(m_numLeaves);

        buildFloat(leafBuffers, leafManager);
        buildVector(leafBuffers, leafManager);

        ScopedTimer timer("IndexGridBuilder::build");


        parallel_for(tbb::blocked_range<size_t>(0, m_numLeaves), [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                const auto& leaf = leafManager.leaf(i);
                size_t localIdx = 0;

                for (auto iter = leaf.cbeginValueOn(); iter.test(); ++iter) {
                    const size_t outIdx = m_leafOffsets[i] + localIdx;
                    m_outData->pIndexes()[outIdx] = outIdx;
                    m_outData->pCoords()[outIdx] = iter.getCoord();
                    ++localIdx;
                }
            }
        });

        // Process leaves in parallel
        parallel_for(tbb::blocked_range<size_t>(0, m_numLeaves), [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                const auto& leaf = leafManager.leaf(i);

                // Copy value buffers directly using memcpy for active values
                for (const BufferInfo& bufferInfo : leafBuffers[i]) {
                    const auto& valMask = leaf.valueMask();
                    size_t activeIdx = 0;

                    for (size_t voxelIdx = 0; voxelIdx < TreeType::LeafNodeType::SIZE; ++voxelIdx) {
                        if (valMask.isOn(voxelIdx)) {
                            char* dest = static_cast<char*>(bufferInfo.destPtr) + activeIdx * bufferInfo.elementSize;
                            const char* src = static_cast<const char*>(bufferInfo.sourceBuffer) + voxelIdx * bufferInfo.elementSize;
                            std::memcpy(dest, src, bufferInfo.elementSize);
                            ++activeIdx;
                        }
                    }
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
        leafManager.foreach ([&data, &domainLeafManager, this](LeafT& leaf, const size_t leafIdx) {
            const size_t leafBaseOffset = m_leafOffsets[leafIdx];
            const auto& valueMask = leaf.getValueMask();
            ValueT* leafData = leaf.buffer().data();

            if (valueMask.isOn()) {
                std::memcpy(leafData, data + leafBaseOffset, LeafT::SIZE * sizeof(ValueT));
            } else {
                size_t localIndex = 0;
                for (auto iter = valueMask.beginOn(); iter; ++iter) {
                    leafData[iter.pos()] = data[leafBaseOffset + localIndex];
                    ++localIndex;
                }
            }
        });

        grid->pruneGrid();
        return grid;
    }

    void setAllocType(const AllocationType type) const { m_outData->setAllocationType(type); }

   private:
    void computeLeafs() {
        ScopedTimer timer("IndexGridBuilder::computeLeafs");
        using TreeType = typename DomainGridT::TreeType;
        const TreeType& tree = m_domainGrid->tree();
        const openvdb::tree::LeafManager<const TreeType> leafManager(tree);
        m_numLeaves = leafManager.leafCount();

        m_leafVoxelCounts.resize(m_numLeaves);
        m_leafOffsets.resize(m_numLeaves + 1);

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
            [&](const tbb::blocked_range<size_t>& r, size_t sum, const bool isFinal) -> size_t {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    if (isFinal) {
                        m_leafOffsets[i] = sum;
                    }
                    sum += m_leafVoxelCounts[i];
                }
                return sum;
            },
            std::plus<>());

        m_leafOffsets[m_numLeaves] = m_totalVoxels;
    }

    typename DomainGridT::ConstPtr m_domainGrid;
    std::vector<size_t> m_leafVoxelCounts{};
    std::vector<size_t> m_leafOffsets{};
    size_t m_totalVoxels = 0;
    size_t m_numLeaves = 0;
    GridIndexedData* m_outData;

    std::unordered_map<std::string, openvdb::FloatGrid::Ptr> m_float_grids;
    std::unordered_map<std::string, openvdb::VectorGrid::Ptr> m_vector_grids;
};

*/

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

	out_data.addValueBlock<float>(grid->getName(), totalVoxels);
	out_data.addValueBlock<openvdb::Vec3f>(domain->getName(), totalVoxels);
	out_data.allocateCoords(totalVoxels);

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
}  // namespace HNS