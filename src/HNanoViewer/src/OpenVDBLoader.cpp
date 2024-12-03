//
// Created by zphrfx on 02/12/2024.
//

#include "OpenVDBLoader.hpp"
#include <Utils/OpenToNano.hpp>
#include <iostream>


OpenVDBLoader::OpenVDBLoader() { initialize(); }

OpenVDBLoader::~OpenVDBLoader() { shutdown(); }

void OpenVDBLoader::initialize() { openvdb::initialize(); }

void OpenVDBLoader::shutdown() { openvdb::uninitialize(); }

bool OpenVDBLoader::loadVDBToTexture(const std::string& filename, GLuint& volumeTexture, glm::vec3& volumeDimensions)
{
    // Open the VDB file
    openvdb::io::File file(filename);

    try {
        file.open();
    } catch (openvdb::IoError& e) {
        std::cerr << "OpenVDB Error: " << e.what() << std::endl;
        return false;
    }

    // Read the first grid in the file
    const openvdb::GridBase::Ptr baseGrid = file.readGrid(file.beginName().gridName());

    // Check if it's a FloatGrid
	const openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (!grid) {
        std::cerr << "Error: Grid is not a FloatGrid." << std::endl;
        return false;
    }

    // Determine volume bounds and dimensions
    openvdb::math::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    openvdb::Coord minCoord = bbox.min();
    openvdb::Coord maxCoord = bbox.max();

	const int dimX = maxCoord.x() - minCoord.x() + 1;
    const int dimY = maxCoord.y() - minCoord.y() + 1;
    const int dimZ = maxCoord.z() - minCoord.z() + 1;

    volumeDimensions = glm::vec3(dimX, dimY, dimZ);

    // Create and fill the volume data
	const auto volumeData = static_cast<float*>(malloc(dimX * dimY * dimZ * sizeof(float)));

    for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
		const float value = *iter;

        const int x = coord.x() - minCoord.x();
        const int y = coord.y() - minCoord.y();
        const int z = coord.z() - minCoord.z();

        const int index = x + y * dimX + z * dimX * dimY;

        volumeData[index] = value;
    }

    // Create OpenGL 3D texture
    glGenTextures(1, &volumeTexture);
    glBindTexture(GL_TEXTURE_3D, volumeTexture);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, dimX, dimY, dimZ, 0, GL_RED, GL_FLOAT, &volumeData[0]);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	constexpr float borderColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor);

    file.close();

	free(volumeData);

    return true;
}
