//
// Created by zphrfx on 05/09/2024.
//

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tree/NodeManager.h>

#include <memory>

#include "GridData.hpp"

// iterate over openvdb grid
// launch a kernel with the [pos, value] pairs
// use cudaVoxelToGrid to write to nanovdb grid
// ... calculations
// write back to a [pos, value] to build openvdb grid

// see : https://github.com/danrbailey/siggraph2023_openvdb/blob/master/benchmarks/cloud_value_clamp/main.cpp
// for Multi-threaded OpenVDB iteration

namespace HNS {
template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
	static constexpr std::size_t alignment = Alignment;
	AlignedAllocator() noexcept = default;

	template <typename U>
	explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

	static T* Allocate(const std::size_t n) {
		void* ptr = _aligned_malloc(n * sizeof(T), alignment);
		if (!ptr) throw std::bad_alloc();

		return static_cast<T*>(ptr);
	}

	static void Free(T* const ptr) noexcept { _aligned_free(ptr); };
};

template <typename CoordT, typename ValueT>
struct alignas(64) ThreadBuffer {
	static constexpr size_t CHUNK_SIZE = 1024;

	using AlignedCoordVec = std::vector<CoordT, AlignedAllocator<CoordT, 64>>;
	using AlignedValueVec = std::vector<ValueT, AlignedAllocator<ValueT, 64>>;

	AlignedCoordVec coords;
	AlignedValueVec values;
	size_t writeOffset;  // Pre-calculated write position

	ThreadBuffer() : writeOffset(0) {
		coords.reserve(CHUNK_SIZE);
		values.reserve(CHUNK_SIZE);
	}
};

template <typename T>
struct SimdCopy {
	static void copy(T* dst, const T* src, size_t count) {
		if constexpr (sizeof(T) % 32 == 0) {
			const auto src_avx = reinterpret_cast<const __m256i*>(src);
			const auto dst_avx = reinterpret_cast<__m256i*>(dst);
			const size_t avx_count = (count * sizeof(T)) / 32;

			for (size_t i = 0; i < avx_count; ++i) {
				_mm256_stream_si256(&dst_avx[i], _mm256_load_si256(&src_avx[i]));
			}
			_mm_sfence();
		} else if constexpr (sizeof(T) % 16 == 0) {
			__movsq(reinterpret_cast<unsigned __int64*>(dst), reinterpret_cast<const unsigned __int64*>(src), (count * sizeof(T)) / 8);
		} else {
			memcpy(dst, src, count * sizeof(T));
		}
	}
};

struct NodeInfo {
	size_t voxelCount;
	size_t startOffset;
};

template <typename GridT, typename CoordT, typename ValueT>
void extractFromOpenVDB_v2(const typename GridT::ConstPtr& grid, GridData<CoordT, ValueT>& out_data) {
	const auto& tree = grid->tree();
	out_data.size = tree.activeVoxelCount();

	// Aligned memory allocation
	out_data.pCoords = new CoordT[out_data.size];
	out_data.pValues = new ValueT[out_data.size];

	if (!out_data.pCoords || !out_data.pValues) {
		if (out_data.pCoords) _aligned_free(out_data.pCoords);
		if (out_data.pValues) _aligned_free(out_data.pValues);
		throw std::bad_alloc();
	}

	// First pass: count voxels in each node and calculate offsets
	std::vector<NodeInfo> nodeInfos;
	nodeInfos.reserve(tree.nodeCount()[0] + tree.nodeCount()[1] + tree.nodeCount()[2]);

	size_t currentOffset = 0;
	openvdb::tree::NodeManager<const typename GridT::TreeType> nodeManager(tree);

	// Calculate offsets
	nodeManager.foreachTopDown([&](const auto& node) {
		size_t voxelCount = 0;
		for (auto iter = node.cbeginValueOn(); iter; ++iter) {
			++voxelCount;
		}

		if (voxelCount > 0) {
			nodeInfos.push_back({voxelCount, currentOffset});
			currentOffset += voxelCount;
		}
	});

	// Second pass: extract data using pre-calculated offsets using TBB
	tbb::parallel_for(tbb::blocked_range<size_t>(0, nodeInfos.size()), [&](const tbb::blocked_range<size_t>& range) {
		// Thread-local buffer
		static thread_local ThreadBuffer<CoordT, ValueT> buffer;

		for (size_t i = range.begin(); i != range.end(); ++i) {
			const auto& [voxelCount, startOffset] = nodeInfos[i];
			buffer.writeOffset = startOffset;
			buffer.coords.clear();
			buffer.values.clear();

			size_t localOffset = 0;
			auto node = tree.beginNode()->getChild(i);  // Get node by index

			for (auto iter = node.cbeginValueOn(); iter; ++iter) {
				buffer.coords.push_back(iter.getCoord());
				buffer.values.push_back(iter.getValue());

				if (buffer.coords.size() >= ThreadBuffer<CoordT, ValueT>::CHUNK_SIZE) {
					// Direct copy to pre-calculated position
					SimdCopy<CoordT>::copy(out_data.pCoords + buffer.writeOffset + localOffset, buffer.coords.data(), buffer.coords.size());
					SimdCopy<ValueT>::copy(out_data.pValues + buffer.writeOffset + localOffset, buffer.values.data(), buffer.values.size());

					localOffset += buffer.coords.size();
					buffer.coords.clear();
					buffer.values.clear();
				}
			}

			// Copy remaining data
			if (!buffer.coords.empty()) {
				SimdCopy<CoordT>::copy(out_data.pCoords + buffer.writeOffset + localOffset, buffer.coords.data(), buffer.coords.size());
				SimdCopy<ValueT>::copy(out_data.pValues + buffer.writeOffset + localOffset, buffer.values.data(), buffer.values.size());
			}
		}
	});
}


template <typename GridT, typename ValueT>
void extractFromOpenVDB(const typename GridT::ConstPtr& grid, OpenGrid<ValueT>& out_data) {
	const auto& tree = grid->tree();

	out_data.allocateStandard(tree.activeVoxelCount());

	std::atomic<size_t> writePos{0};
	constexpr size_t chunkSize = 1024;

	struct ThreadLocalBuffers {
		std::vector<openvdb::Coord> coords;
		std::vector<ValueT> values;
	};

	thread_local ThreadLocalBuffers buffers;
	openvdb::tree::NodeManager<const typename GridT::TreeType> nodeManager(tree);

	// Enable parallel execution if available
	nodeManager.foreachTopDown([&](const auto& node) {
		auto& localCoords = buffers.coords;
		auto& localValues = buffers.values;

		localCoords.clear();
		localValues.clear();
		if (localCoords.capacity() < chunkSize) {
			localCoords.reserve(chunkSize);
			localValues.reserve(chunkSize);
		}

		auto iter = node.cbeginValueOn();
		while (iter) {
			size_t count = 0;
			do {
				localCoords.push_back(iter.getCoord());
				localValues.push_back(iter.getValue());
				++iter;
				++count;
			} while (count < chunkSize && iter);

			if (count > 0) {
				const size_t pos = writePos.fetch_add(count);
				// Use MSVC intrinsics for memory copy if available
				if constexpr (sizeof(openvdb::Coord) % 16 == 0) {
					__movsq(reinterpret_cast<unsigned __int64*>(out_data.pCoords() + pos),
					        reinterpret_cast<const unsigned __int64*>(localCoords.data()), (count * sizeof(openvdb::Coord)) / 8);
				} else {
					memcpy(out_data.pCoords() + pos, localCoords.data(), count * sizeof(openvdb::Coord));
				}

				if constexpr (sizeof(ValueT) % 16 == 0) {
					__movsq(reinterpret_cast<unsigned __int64*>(out_data.pValues() + pos),
					        reinterpret_cast<const unsigned __int64*>(localValues.data()), (count * sizeof(ValueT)) / 8);
				} else {
					memcpy(out_data.pValues() + pos, localValues.data(), count * sizeof(ValueT));
				}
			}
		}
	});
}
}  // namespace HNS