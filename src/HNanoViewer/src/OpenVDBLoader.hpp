//
// Created by zphrfx on 02/12/2024.
//

#pragma once

#include <glad/glad.h>
#include <openvdb/openvdb.h>

#include <glm/glm.hpp>
#include <string>

class OpenVDBLoader {
   public:
	OpenVDBLoader();
	~OpenVDBLoader();

	// Load VDB file and extract voxel positions
	static bool loadVDBToTexture(const std::string& filename, GLuint& volumeTexture, glm::vec3& volumeDimensions);

   private:
	// Initialize OpenVDB library (called in constructor)
	static void initialize();

	// Shutdown OpenVDB library (called in destructor)
	static void shutdown();

};
