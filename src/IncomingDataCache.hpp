//
// Created by zphrfx on 06/08/2024.
//

#pragma once

#include <openvdb/openvdb.h>

class IncomingDataCache {
public:
	IncomingDataCache() = default;
	~IncomingDataCache() = default;
	// remove copy semantics
	IncomingDataCache(const IncomingDataCache&) = delete;
	IncomingDataCache& operator=(const IncomingDataCache&) = delete;


	void setDensity(const openvdb::FloatGrid::Ptr& pDensity) { pDensityGrid = pDensity; }
	void setVelocity(const openvdb::VectorGrid::Ptr& pVelocity) { pVelocityGrid = pVelocity; }
	[[nodiscard]] openvdb::FloatGrid::Ptr getDensityCached() const { return pDensityGrid; }
	[[nodiscard]] openvdb::VectorGrid::Ptr getVelocityCached() const { return pVelocityGrid; }

private:
	openvdb::FloatGrid::Ptr pDensityGrid = nullptr;
	openvdb::VectorGrid::Ptr pVelocityGrid = nullptr;
};