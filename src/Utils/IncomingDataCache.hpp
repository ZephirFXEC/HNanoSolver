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

class IncomingDataCache {
   public:
	IncomingDataCache() = default;
	~IncomingDataCache() { densityHandle.reset(); }

	// Remove copy semantics
	IncomingDataCache(const IncomingDataCache&) = delete;
	IncomingDataCache& operator=(const IncomingDataCache&) = delete;

	// Allow move semantics
	IncomingDataCache(IncomingDataCache&&) = default;
	IncomingDataCache& operator=(IncomingDataCache&&) = default;

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& getCachedDensityGrid(const openvdb::FloatGrid::ConstPtr& densGrid) {
		if (!densityHandle) {
			densityHandle = std::make_unique<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>>(
			    nanovdb::createNanoGrid<openvdb::FloatGrid, float, nanovdb::CudaDeviceBuffer>(*densGrid));
			printf("Creating NanoVDB density grid\n");
		}
		return *densityHandle;
	}


	void swap(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& other) {
		if (densityHandle) {
			std::swap(*densityHandle, other);
		} else {
			densityHandle = std::make_unique<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>>(std::move(other));
		}
	}


	[[nodiscard]] bool hasDensityGrid() const { return densityHandle != nullptr; }

	void Invalidate();

   private:
	std::unique_ptr<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>> densityHandle = nullptr;
};

#endif  // __INCOMINGDATACACHE_HPP__