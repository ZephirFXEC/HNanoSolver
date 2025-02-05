//
// Created by zphrfx on 25/01/2025.
//

#define NANOVDB_USE_OPENVDB

#include <gtest/gtest.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <openvdb/openvdb.h>

#include "Utils/GridBuilder.hpp"
#include "Utils/GridData.hpp"
#include "Utils/Stencils.hpp"


struct Vec3f {
	float x, y, z;
};

TEST(GridIndexedDataTest, IndexGridModification) {
	openvdb::FloatGrid::Ptr denPtr = openvdb::FloatGrid::create();
	denPtr->setName("density");
	denPtr->setGridClass(openvdb::GRID_FOG_VOLUME);
	denPtr->setTransform(openvdb::math::Transform::createLinearTransform(1.0f));

	openvdb::VectorGrid::Ptr velPtr = openvdb::VectorGrid::create();
	velPtr->setName("velocity");
	velPtr->setGridClass(openvdb::GRID_STAGGERED);


	for (int i = 0; i < 15; ++i) {
		denPtr->getAccessor().setValue(openvdb::Coord(i, 0, 0), i);
		velPtr->getAccessor().setValue(openvdb::Coord(0, i, 0), openvdb::Vec3f(i, i, i));
	}

	EXPECT_EQ(velPtr->activeVoxelCount(), 15);
	EXPECT_EQ(denPtr->activeVoxelCount(), 15);

	openvdb::FloatGrid::Ptr domain = openvdb::FloatGrid::create();
	domain->topologyUnion(*denPtr);
	domain->topologyUnion(*velPtr);

	HNS::GridIndexedData indexed_data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &indexed_data);

	builder.addGrid(denPtr, "density");
	builder.addGrid(velPtr, "velocity");

	builder.build();

	auto* densityPtr = indexed_data.pValues<float>("density");
	ASSERT_NE(densityPtr, nullptr);

	auto* velocityPtr = indexed_data.pValues<openvdb::Vec3f>("velocity");
	ASSERT_NE(velocityPtr, nullptr);

	using SrcGridT = openvdb::FloatGrid;
	using DstBuildT = nanovdb::ValueOnIndex;
	using BufferT = nanovdb::HostBuffer;

	nanovdb::GridHandle<BufferT> idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*domain, 1u, false, false, 0);
	auto indexGrid = idxHandle.grid<DstBuildT>();

	IndexOffsetSampler<0> idxSampler(*indexGrid);
	IndexSampler<float, 1> custom_trilinear_f(idxSampler, densityPtr);
	IndexSampler<openvdb::Vec3f, 1> custom_trilinear_v(idxSampler, velocityPtr);


	for (int i = 0; i < indexed_data.size(); ++i) {
		densityPtr[i] *= 2.0f;
		velocityPtr[i] *= 2.0f;
	}

	EXPECT_FLOAT_EQ(custom_trilinear_f(nanovdb::Coord(5, 0, 0)), 10);
	EXPECT_FLOAT_EQ(custom_trilinear_v(nanovdb::Coord(0, 5, 0))[0], 10);

	EXPECT_FLOAT_EQ(custom_trilinear_f(nanovdb::Vec3f(5.5, 0, 0)), 11);
	EXPECT_FLOAT_EQ(custom_trilinear_v(nanovdb::Vec3f(0, 5.5, 0))[0], 11);

	auto dengridptr = builder.writeIndexGrid<openvdb::FloatGrid>("density", domain->voxelSize()[0]);
	auto acc_f = dengridptr->getAccessor();

	EXPECT_FLOAT_EQ(acc_f.getValue(openvdb::Coord(5, 0, 0)), 10);
	EXPECT_FLOAT_EQ(acc_f.getValue(openvdb::Coord(6, 0, 0)), 12);

	auto velgridptr = builder.writeIndexGrid<openvdb::VectorGrid>("velocity", domain->voxelSize()[0]);
	auto acc_v = velgridptr->getAccessor();

	EXPECT_FLOAT_EQ(acc_v.getValue(openvdb::Coord(0, 5, 0))[0], 10);
	EXPECT_FLOAT_EQ(acc_v.getValue(openvdb::Coord(0, 6, 0))[0], 12);
}


TEST(GridIndexedDataTest, TrilinearSampler) {
	openvdb::FloatGrid::Ptr denPtr = openvdb::FloatGrid::create();
	denPtr->setName("density");
	denPtr->setGridClass(openvdb::GRID_FOG_VOLUME);
	denPtr->setTransform(openvdb::math::Transform::createLinearTransform(1.0f));

	openvdb::VectorGrid::Ptr velPtr = openvdb::VectorGrid::create();
	velPtr->setName("velocity");
	velPtr->setGridClass(openvdb::GRID_STAGGERED);


	for (size_t i = 0; i < 15; ++i) {
		denPtr->getAccessor().setValue(openvdb::Coord(i, 0, 0), i);
		velPtr->getAccessor().setValue(openvdb::Coord(0, i, 0), openvdb::Vec3f(i, i, i));
	}

	EXPECT_EQ(velPtr->activeVoxelCount(), 15);
	EXPECT_EQ(denPtr->activeVoxelCount(), 15);

	openvdb::FloatGrid::Ptr domain = openvdb::FloatGrid::create();
	domain->topologyUnion(*denPtr);
	domain->topologyUnion(*velPtr);

	HNS::GridIndexedData indexed_data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &indexed_data);

	builder.addGrid(denPtr, "density");
	builder.addGrid(velPtr, "velocity");

	builder.build();

	auto* densityBlock = indexed_data.getValueBlock<float>("density");
	ASSERT_NE(densityBlock, nullptr);
	auto* densityPtr = indexed_data.pValues<float>("density");
	ASSERT_NE(densityPtr, nullptr);

	auto* velocityBlock = indexed_data.getValueBlock<openvdb::Vec3f>("velocity");
	ASSERT_NE(velocityBlock, nullptr);
	auto* velocityPtr = indexed_data.pValues<openvdb::Vec3f>("velocity");
	ASSERT_NE(velocityPtr, nullptr);

	using SrcGridT = openvdb::FloatGrid;
	using DstBuildT = nanovdb::ValueOnIndex;
	using BufferT = nanovdb::HostBuffer;

	nanovdb::GridHandle<BufferT> idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*domain, 1u, false, false, 0);
	auto indexGrid = idxHandle.grid<DstBuildT>();

	IndexOffsetSampler<0> idxSampler(*indexGrid);
	IndexSampler<float, 1> custom_trilinear_f(idxSampler, densityPtr);
	IndexSampler<openvdb::Vec3f, 1> custom_trilinear_v(idxSampler, velocityPtr);

	EXPECT_FLOAT_EQ(custom_trilinear_f(nanovdb::Coord(0, 0, 0)), 0);
	EXPECT_FLOAT_EQ(custom_trilinear_f(nanovdb::Coord(1, 0, 0)), 1);

	EXPECT_FLOAT_EQ(custom_trilinear_f(nanovdb::Coord(1, 1, 0)), 0);
	EXPECT_FLOAT_EQ(custom_trilinear_v(nanovdb::Coord(1, 1, 0))[0], 0);

	EXPECT_FLOAT_EQ(custom_trilinear_v(nanovdb::Coord(0, 0, 0))[0], 0);
	EXPECT_FLOAT_EQ(custom_trilinear_v(nanovdb::Coord(0, 1, 0))[0], 1);


	EXPECT_FLOAT_EQ(custom_trilinear_f(nanovdb::Vec3f(0.5, 0, 0)), 0.5);
	EXPECT_FLOAT_EQ(custom_trilinear_f(nanovdb::Vec3f(1.25, 0, 0)), 1.25);

	EXPECT_FLOAT_EQ(custom_trilinear_v(nanovdb::Vec3f(0, 0.5, 0))[0], 0.5);
	EXPECT_FLOAT_EQ(custom_trilinear_v(nanovdb::Vec3f(0, 1.25, 0))[0], 1.25);
}


TEST(GridIndexedDataTest, IndexSampler) {
	openvdb::FloatGrid::Ptr gridPtr = openvdb::FloatGrid::create();
	gridPtr->setName("density");
	gridPtr->setGridClass(openvdb::GRID_FOG_VOLUME);
	gridPtr->setTransform(openvdb::math::Transform::createLinearTransform(1.0f));


	for (size_t i = 0; i < 15; ++i) {
		gridPtr->getAccessor().setValue(openvdb::Coord(i, 0, 0), i);
	}

	EXPECT_EQ(gridPtr->activeVoxelCount(), 15);


	openvdb::VectorGrid::Ptr domain = openvdb::VectorGrid::create();
	domain->setName("velocity");
	gridPtr->setGridClass(openvdb::GRID_STAGGERED);

	auto domain_acc = domain->getAccessor();
	int i = 0;
	for (auto iter = gridPtr->cbeginValueOn(); iter; ++iter) {
		auto coord = iter.getCoord();

		// for each coord set the neighboring coord too
		const auto left = coord + openvdb::Coord(-1, 0, 0);
		const auto right = coord + openvdb::Coord(1, 0, 0);

		domain_acc.setValue(coord, openvdb::Vec3f(i, i, i));
		domain_acc.setValue(left, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		domain_acc.setValue(right, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		i++;
	}


	HNS::GridIndexedData indexed_data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(gridPtr, &indexed_data);

	builder.addGrid(gridPtr, "density");
	builder.addGrid(domain, "velocity");

	builder.build();

	using SrcGridT = openvdb::FloatGrid;
	using DstBuildT = nanovdb::ValueOnIndex;
	using BufferT = nanovdb::HostBuffer;

	nanovdb::GridHandle<BufferT> idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*gridPtr, 1u, false, false, 0);
	auto indexGrid = idxHandle.grid<DstBuildT>();

	auto dataf = indexed_data.pValues<float>("density");
	auto datav = indexed_data.pValues<openvdb::Vec3f>("velocity");

	IndexOffsetSampler<0> idxSampler(*indexGrid);

	IndexSampler<float, 0> samplerf(idxSampler, dataf);
	IndexSampler<openvdb::Vec3f, 0> samplerv(idxSampler, datav);

	nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> chanAccess(*indexGrid);

	for (size_t i = 0; i < indexed_data.size(); ++i) {
		auto sampler_data_f = samplerf(nanovdb::Coord(i, 0, 0));
		auto sampler_data_v = samplerv(nanovdb::Coord(i, 0, 0));
		auto nanosampler = chanAccess(i, 0, 0);

		EXPECT_EQ(sampler_data_f, nanosampler);
		EXPECT_EQ(datav[i].x(), sampler_data_v[0]);
	}

	EXPECT_EQ(samplerf(nanovdb::Coord(-10, -10, -10)), 0);
}


TEST(GridIndexedDataTest, IndexGridBuilder) {
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


	HNS::GridIndexedData indexed_data;
	HNS::IndexGridBuilder<openvdb::VectorGrid> builder(domain, &indexed_data);

	builder.addGrid(gridPtr, "density");
	builder.addGrid(domain, "velocity");

	builder.build();


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
		uint64_t idx = indexed_data.pIndexes()[i];

		float density = densityPtr[idx];
		openvdb::Vec3f velocity = velocityPtr[idx];

		if (density == 420) {
			EXPECT_EQ(density, velocity[0]);
		}
	}
}

TEST(GridIndexedDataTest, ExtractFromVDB) {
	HNS::GridIndexedData indexed_data;

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
		uint64_t idx = indexed_data.pIndexes()[i];

		float density = densityPtr[idx];
		openvdb::Vec3f velocity = velocityPtr[idx];

		if (density == 420) {
			EXPECT_EQ(density, velocity[0]);
		}
	}
}

TEST(GridIndexedDataTest, AllocateCoordsAndAddBlocks) {
	HNS::GridIndexedData grid;

	constexpr size_t numElements = 5;
	bool success = grid.allocateCoords(numElements, AllocationType::Standard);
	EXPECT_TRUE(success);
	EXPECT_EQ(grid.size(), numElements);

	auto* idx = grid.pIndexes();
	ASSERT_NE(idx, nullptr);
	for (size_t i = 0; i < numElements; ++i) {
		idx[i] = 100 + i;
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
		EXPECT_EQ(idx[i], 100 + i);

		EXPECT_FLOAT_EQ(densityPtr[i], static_cast<float>(i) * 1.1f);

		EXPECT_FLOAT_EQ(velocityPtr[i].x, static_cast<float>(i) + 0.1f);
		EXPECT_FLOAT_EQ(velocityPtr[i].y, static_cast<float>(i) + 0.2f);
		EXPECT_FLOAT_EQ(velocityPtr[i].z, static_cast<float>(i) + 0.3f);
	}
}

TEST(GridIndexedDataTest, ClearBlocks) {
	HNS::GridIndexedData grid;
	constexpr size_t numElements = 3;
	bool success = grid.allocateCoords(numElements, AllocationType::Standard);
	EXPECT_TRUE(success);

	success = grid.addValueBlock<float>("density", AllocationType::Standard, numElements);
	EXPECT_TRUE(success);

	auto* idx = grid.pIndexes();
	auto* density = grid.pValues<float>("density");
	for (size_t i = 0; i < numElements; ++i) {
		idx[i] = i;
		density[i] = static_cast<float>(i);
	}

	grid.clearValues();
	EXPECT_EQ(grid.numValueBlocks(), static_cast<size_t>(0));
	EXPECT_EQ(grid.size(), numElements);

	grid.clear();
	EXPECT_EQ(grid.size(), static_cast<size_t>(0));
	EXPECT_EQ(grid.numValueBlocks(), static_cast<size_t>(0));
}
