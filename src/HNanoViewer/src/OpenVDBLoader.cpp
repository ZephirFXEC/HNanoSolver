//
// Created by zphrfx on 02/12/2024.
//

#include "OpenVDBLoader.hpp"

#include <nanovdb/math/Math.h>

#include <Utils/GridBuilder.hpp>
#include <iostream>


OpenVDBLoader::OpenVDBLoader() { initialize(); }

OpenVDBLoader::~OpenVDBLoader() { shutdown(); }

void OpenVDBLoader::initialize() { openvdb::initialize(); }

void OpenVDBLoader::shutdown() { openvdb::uninitialize(); }

bool OpenVDBLoader::loadVDB(const std::string& filename) {
	// Open the VDB file
	openvdb::io::File file(filename);

	try {
		file.open();
	} catch (openvdb::IoError& e) {
		std::cerr << "OpenVDB Error: " << e.what() << std::endl;
		return false;
	}

	// Read the first grid in the file
	pBaseGrid = file.readGrid(file.beginName().gridName());

	// Check if it's a FloatGrid
	if (const openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(pBaseGrid); !grid) {
		std::cerr << "Error: Grid is not a FloatGrid." << std::endl;
		return false;
	}

	file.close();

	return true;
}

std::vector<std::pair<nanovdb::Coord, float>> OpenVDBLoader::getCoords() const {
	const openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(pBaseGrid);
	const auto accessor = grid->getAccessor();

	std::vector<std::pair<nanovdb::Coord, float>> coords(grid->activeVoxelCount());
	openvdb::Coord minbbox = grid->evalActiveVoxelBoundingBox().min();

	for (auto iter = grid->tree().beginValueOn(); iter.test(); ++iter) {
		const openvdb::Coord coord = iter.getCoord();
		const float value = iter.getValue();

		openvdb::Coord toplus = coord - minbbox;
		nanovdb::Coord pCoords(toplus.x(), toplus.y(), toplus.z());

		coords.emplace_back(pCoords, value);
	}

	return coords;
}

bool OpenVDBLoader::VDBToTexture(GLuint& volumeTexture, HNS::GridIndexedData* in_data, openvdb::math::BBox<openvdb::Vec3d>& bbox) const {
	// Determine volume bounds and dimensions
	const openvdb::FloatGrid::ConstPtr grid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(pBaseGrid);

	openvdb::math::CoordBBox box = grid->evalActiveVoxelBoundingBox();
	openvdb::Coord minCoord = box.min();
	openvdb::Coord maxCoord = box.max();
	bbox = grid->transform().indexToWorld(box);

	const int dimX = maxCoord.x() - minCoord.x() + 1;
	const int dimY = maxCoord.y() - minCoord.y() + 1;
	const int dimZ = maxCoord.z() - minCoord.z() + 1;

	// Create and fill the volume data
	// Pre-allocate with zero initialization
	std::vector volumeData(dimX * dimY * dimZ, 0.0f);

	const auto* pCoords = in_data->pCoords();
	const auto* pValues = in_data->pValues<float>("density");

	// Use direct iteration over coordinates with pre-computed index calculation
	for (size_t i = 0; i < in_data->size(); ++i) {
		const openvdb::Coord& coord = pCoords[i];
		const int x = coord.x() - minCoord.x();
		const int y = coord.y() - minCoord.y();
		const int z = coord.z() - minCoord.z();

		volumeData[x + y * dimX + z * dimX * dimY] = pValues[i];
	}

	// Create OpenGL 3D texture
	glGenTextures(1, &volumeTexture);
	glBindTexture(GL_TEXTURE_3D, volumeTexture);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, dimX, dimY, dimZ, 0, GL_RED, GL_FLOAT, volumeData.data());

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	constexpr float borderColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
	glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor);

	return true;
}
