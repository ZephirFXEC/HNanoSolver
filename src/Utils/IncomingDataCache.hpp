//
// Created by zphrfx on 06/08/2024.
//

#ifndef __INCOMINGDATACACHE_HPP__
#define __INCOMINGDATACACHE_HPP__

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <openvdb/openvdb.h>
#include <nanovdb/tools/CreateNanoGrid.h>

class IncomingDataCache {
public:
    IncomingDataCache() = default;

    ~IncomingDataCache() = default;

    // Remove copy semantics
    IncomingDataCache(const IncomingDataCache&) = delete;
    IncomingDataCache& operator=(const IncomingDataCache&) = delete;

    // Allow move semantics
    IncomingDataCache(IncomingDataCache&&) = delete;
    IncomingDataCache& operator=(IncomingDataCache&&) = delete;

    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& getCachedDensityGrid(const openvdb::FloatGrid::ConstPtr& densGrid) {
		if (!densityHandle) {
            densityHandle = nanovdb::tools::createNanoGrid<openvdb::FloatGrid, float, nanovdb::cuda::DeviceBuffer>(*densGrid);
			printf("Creating NanoVDB density grid\n");
        }
        return densityHandle;
    }

    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& getCachedVelocityGrid(const openvdb::VectorGrid::ConstPtr& velGrid) {
		if (!velocityHandle) {
            velocityHandle = nanovdb::tools::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::cuda::DeviceBuffer>(*velGrid);
			printf("Creating NanoVDB velocity grid\n");
        }
        return velocityHandle;
    }

private:

    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> densityHandle;
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> velocityHandle;
};

#endif  // __INCOMINGDATACACHE_HPP__