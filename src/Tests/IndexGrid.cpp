//
// Created by zphrfx on 25/01/2025.
//


#include <gtest/gtest.h>
#include <nanovdb/tools/GridBuilder.h>

#include "Utils/GridData.hpp"

struct Vec3f {
	float x, y, z;
};


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
