//
// Created by zphrfx on 06/08/2024.
//

#ifndef __INCOMINGDATACACHE_HPP__
#define __INCOMINGDATACACHE_HPP__

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
#include <openvdb/openvdb.h>
#include <memory>

#include "ScopedTimer.hpp"

class IncomingDataCache {
   public:
	IncomingDataCache() = default;
	~IncomingDataCache() { pHandle.reset(); }

	// Remove copy semantics
	IncomingDataCache(const IncomingDataCache&) = delete;
	IncomingDataCache& operator=(const IncomingDataCache&) = delete;

	// Allow move semantics
	IncomingDataCache(IncomingDataCache&&) = default;
	IncomingDataCache& operator=(IncomingDataCache&&) = default;

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& getCachedGrid(const openvdb::VectorGrid::ConstPtr& grid) {
		if (!pHandle) {
			ScopedTimer timer("Creating NanoVDB Vector grid");
			pHandle = std::make_unique<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>>(
			    nanovdb::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::CudaDeviceBuffer>(*grid));
		}
		return *pHandle;
	}

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& getCachedGrid(const openvdb::FloatGrid::ConstPtr& grid) {
		if (!pHandle) {
			ScopedTimer timer("Creating NanoVDB float grid");
			pHandle = std::make_unique<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>>(
				nanovdb::createNanoGrid<openvdb::FloatGrid, float, nanovdb::CudaDeviceBuffer>(*grid));
		}
		return *pHandle;
	}


	void swap(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& other) {
		if (pHandle) {
			std::swap(*pHandle, other);
		} else {
			pHandle = std::make_unique<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>>(std::move(other));
		}
	}


	[[nodiscard]] bool hasDensityGrid() const { return pHandle != nullptr; }

	void Invalidate() {
		pHandle.reset();
	};

   private:
	std::unique_ptr<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>> pHandle = nullptr;

};

#endif  // __INCOMINGDATACACHE_HPP__