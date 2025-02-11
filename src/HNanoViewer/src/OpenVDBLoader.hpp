//
// Created by zphrfx on 02/12/2024.
//

#pragma once

#include <glad/glad.h>
#include <openvdb/openvdb.h>

#include <glm/glm.hpp>
#include <string>

#include "Utils/GridData.hpp"

class OpenVDBLoader {
   public:
	OpenVDBLoader();
	~OpenVDBLoader();

	[[nodiscard]] auto getGridBase() const { return pBaseGrid; }

	bool loadVDB(const std::string& filename);
	bool VDBToTexture(GLuint& volumeTexture, HNS::GridIndexedData* in_data, openvdb::math::BBox<openvdb::Vec3d>& bbox) const;

   private:
	static void initialize();
	static void shutdown();

	openvdb::GridBase::Ptr pBaseGrid = nullptr;
};
