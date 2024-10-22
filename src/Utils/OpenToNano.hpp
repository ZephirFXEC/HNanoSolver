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

template <typename GridT, typename CoordT, typename ValueT>
void extractFromOpenVDB(const typename GridT::ConstPtr& grid, GridData<CoordT, ValueT>& out_data) {
	const auto& tree = grid->tree();
	out_data.size = tree.activeVoxelCount();

	// Aligned allocation for better memory access patterns
	out_data.pCoords = static_cast<CoordT*>(_aligned_malloc(out_data.size * sizeof(CoordT), 64));
	out_data.pValues = static_cast<ValueT*>(_aligned_malloc(out_data.size * sizeof(ValueT), 64));

	if (!out_data.pCoords || !out_data.pValues) {
		if (out_data.pCoords) _aligned_free(out_data.pCoords);
		if (out_data.pValues) _aligned_free(out_data.pValues);
		throw std::bad_alloc();
	}

	std::atomic<size_t> writePos{0};
	// Increased chunk size for better cache utilization
	constexpr size_t chunkSize = 1024;

	// Thread-local buffers with MSVC's __declspec(thread)
#pragma warning(push)
#pragma warning(disable : 4324)  // structure padding warning
	struct alignas(64) ThreadLocalBuffers {
		std::vector<CoordT> coords;
		std::vector<ValueT> values;
		char padding[64];  // Prevent false sharing
	};
#pragma warning(pop)

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
				if constexpr (sizeof(CoordT) % 16 == 0) {
					__movsq(reinterpret_cast<unsigned __int64*>(out_data.pCoords + pos),
					        reinterpret_cast<const unsigned __int64*>(localCoords.data()),
					        (count * sizeof(CoordT)) / 8);
				} else {
					memcpy(out_data.pCoords + pos, localCoords.data(), count * sizeof(CoordT));
				}

				if constexpr (sizeof(ValueT) % 16 == 0) {
					__movsq(reinterpret_cast<unsigned __int64*>(out_data.pValues + pos),
					        reinterpret_cast<const unsigned __int64*>(localValues.data()),
					        (count * sizeof(ValueT)) / 8);
				} else {
					memcpy(out_data.pValues + pos, localValues.data(), count * sizeof(ValueT));
				}
			}
		}
	});
}