//
// Created by zphrfx on 25/01/2025.
//


#include <gtest/gtest.h>
#include <openvdb/openvdb.h>

#include "Utils/GridData.hpp"
#include "Utils/OpenToNano.hpp"

struct Vec3f {
	float x, y, z;
};

TEST(GridIndexedDataTest, ExtractFromVDB) {
	HNS::GridIndexedData<uint32_t> indexed_data;

	openvdb::FloatGrid::Ptr gridPtr = openvdb::FloatGrid::create();
	gridPtr->setName("density");
	gridPtr->setGridClass(openvdb::GRID_FOG_VOLUME);
	gridPtr->setTransform(openvdb::math::Transform::createLinearTransform(1.0f));

	/*
	 * -------------
	 * | 0 | 1 | 0 |
	 * | 1 | 1 | 1 |
	 * | 0 | 1 | 0 |
	 * -------------
	 */
	auto acc = gridPtr->getAccessor();
	constexpr auto center = openvdb::Coord(0, 0, 0);
	constexpr auto left = openvdb::Coord(-1, 0, 0);
	constexpr auto right = openvdb::Coord(1, 0, 0);
	constexpr auto up = openvdb::Coord(0, 1, 0);
	constexpr auto down = openvdb::Coord(0, -1, 0);

	constexpr auto far_center = openvdb::Coord(9, 9, 0);
	constexpr auto far_left = openvdb::Coord(8, 9, 0);
	constexpr auto far_right = openvdb::Coord(11, 9, 0);
	constexpr auto far_up = openvdb::Coord(9, 11, 0);
	constexpr auto far_down = openvdb::Coord(9, 8, 0);

	acc.setValue(center, 1.0f);
	acc.setValue(left, 1.0f);
	acc.setValue(right, 1.0f);
	acc.setValue(up, 1.0f);
	acc.setValue(down, 1.0f);

	acc.setValue(far_center, 1.0f);
	acc.setValue(far_left, 1.0f);
	acc.setValue(far_right, 1.0f);
	acc.setValue(far_up, 1.0f);
	acc.setValue(far_down, 1.0f);

	EXPECT_EQ(gridPtr->activeVoxelCount(), 10);

	openvdb::VectorGrid::Ptr domain = openvdb::VectorGrid::create();
	domain->setName("velocity");
	gridPtr->setGridClass(openvdb::GRID_STAGGERED);

	auto domain_acc = domain->getAccessor();
	for (auto iter = gridPtr->cbeginValueOn(); iter; ++iter) {
		auto coord = iter.getCoord();

		// for each coord set the neighboring coord too
		const auto above = coord + openvdb::Coord(0, 1, 0);
		const auto below = coord + openvdb::Coord(0, -1, 0);
		const auto left = coord + openvdb::Coord(-1, 0, 0);
		const auto right = coord + openvdb::Coord(1, 0, 0);

		domain_acc.setValue(coord, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		domain_acc.setValue(above, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		domain_acc.setValue(below, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		domain_acc.setValue(left, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		domain_acc.setValue(right, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
	}

	acc.setValue(center, 420.0f);
	domain_acc.setValue(center, openvdb::Vec3f(420.0f, 420.0f, 420.0f));


	HNS::extractToGlobalIdx(domain, gridPtr, indexed_data);

	EXPECT_EQ(indexed_data.size(), domain->activeVoxelCount());

	auto* densityBlock = indexed_data.getValueBlock<float>("density");
	ASSERT_NE(densityBlock, nullptr);

	EXPECT_EQ(densityBlock->size(), domain->activeVoxelCount());

	auto* densityPtr = indexed_data.pValues<float>("density");
	ASSERT_NE(densityPtr, nullptr);

	auto* random_block = indexed_data.getValueBlock<float>("random");
	EXPECT_EQ(random_block, nullptr);

	auto* velocityBlock = indexed_data.getValueBlock<openvdb::Vec3f>("velocity");
	ASSERT_NE(velocityBlock, nullptr);

	auto* velocityPtr = indexed_data.pValues<openvdb::Vec3f>("velocity");
	ASSERT_NE(velocityPtr, nullptr);

	for (size_t i = 0; i < indexed_data.size(); ++i) {
		uint32_t coord = indexed_data.pCoords()[i];

		float density = densityPtr[coord];
		openvdb::Vec3f velocity = velocityPtr[coord];

		if (density == 420) {
			EXPECT_EQ(density, velocity[0]);
		}
	}
}

TEST(GridIndexedDataTest, AllocateCoordsAndAddBlocks) {
	HNS::GridIndexedData<uint64_t> grid;

	constexpr size_t numElements = 5;
	bool success = grid.allocateCoords(numElements, AllocationType::Standard);
	EXPECT_TRUE(success);
	EXPECT_EQ(grid.size(), numElements);

	auto* coords = grid.pCoords();
	ASSERT_NE(coords, nullptr);
	for (size_t i = 0; i < numElements; ++i) {
		coords[i] = 100 + i;
	}

	success = grid.addValueBlock<float>("density", AllocationType::Standard, numElements);
	EXPECT_TRUE(success);

	auto* densityBlock = grid.getValueBlock<float>("density");
	ASSERT_NE(densityBlock, nullptr);
	EXPECT_EQ(densityBlock->size(), numElements);
	auto* densityPtr = grid.pValues<float>("density");
	ASSERT_NE(densityPtr, nullptr);

	for (size_t i = 0; i < numElements; ++i) {
		densityPtr[i] = static_cast<float>(i) * 1.1f;
	}

	success = grid.addValueBlock<Vec3f>("velocity", AllocationType::Standard, numElements);
	EXPECT_TRUE(success);

	auto* velBlock = grid.getValueBlock<Vec3f>("velocity");
	ASSERT_NE(velBlock, nullptr);
	EXPECT_EQ(velBlock->size(), numElements);

	auto* velocityPtr = grid.pValues<Vec3f>("velocity");
	ASSERT_NE(velocityPtr, nullptr);

	for (size_t i = 0; i < numElements; ++i) {
		velocityPtr[i].x = static_cast<float>(i) + 0.1f;
		velocityPtr[i].y = static_cast<float>(i) + 0.2f;
		velocityPtr[i].z = static_cast<float>(i) + 0.3f;
	}

	for (size_t i = 0; i < numElements; ++i) {
		EXPECT_EQ(coords[i], 100 + i);

		EXPECT_FLOAT_EQ(densityPtr[i], static_cast<float>(i) * 1.1f);

		EXPECT_FLOAT_EQ(velocityPtr[i].x, static_cast<float>(i) + 0.1f);
		EXPECT_FLOAT_EQ(velocityPtr[i].y, static_cast<float>(i) + 0.2f);
		EXPECT_FLOAT_EQ(velocityPtr[i].z, static_cast<float>(i) + 0.3f);
	}
}

TEST(GridIndexedDataTest, ClearBlocks) {
	HNS::GridIndexedData<uint64_t> grid;
	constexpr size_t numElements = 3;
	bool success = grid.allocateCoords(numElements, AllocationType::Standard);
	EXPECT_TRUE(success);

	success = grid.addValueBlock<float>("density", AllocationType::Standard, numElements);
	EXPECT_TRUE(success);

	auto* coords = grid.pCoords();
	auto* density = grid.pValues<float>("density");
	for (size_t i = 0; i < numElements; ++i) {
		coords[i] = i;
		density[i] = static_cast<float>(i);
	}

	grid.clearValues();
	EXPECT_EQ(grid.numValueBlocks(), static_cast<size_t>(0));
	EXPECT_EQ(grid.size(), numElements);

	grid.clear();
	EXPECT_EQ(grid.size(), static_cast<size_t>(0));
	EXPECT_EQ(grid.numValueBlocks(), static_cast<size_t>(0));
}
