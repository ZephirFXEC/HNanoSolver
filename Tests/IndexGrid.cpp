//
// Created by zphrfx on 25/01/2025.
//

#define NANOVDB_USE_OPENVDB

#include <gtest/gtest.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetUtil.h>

#include "../src/Utils/GridBuilder.hpp"
#include "../src/Utils/GridData.hpp"
#include "../src/Utils/Stencils.hpp"
#include "nanovdb/tools/CreateNanoGrid.h"


struct Vec3f {
	float x, y, z;
};


template <class GridType>
typename GridType::Ptr makeSphere(const float radius, const openvdb::Vec3f& c) {
	typename GridType::Ptr grid = GridType::create();
	using ValueT = typename GridType::ValueType;
	const ValueT outside = grid->background();
	const ValueT inside = -outside;
	const int padding = static_cast<int>(openvdb::math::RoundUp(openvdb::math::Abs(outside)));
	const int dim = static_cast<int>(radius + padding);
	typename GridType::Accessor accessor = grid->getAccessor();
	openvdb::Coord ijk;
	int &i = ijk[0], &j = ijk[1], &k = ijk[2];
	for (i = c[0] - dim; i < c[0] + dim; ++i) {
		const float x2 = openvdb::math::Pow2(i - c[0]);
		for (j = c[1] - dim; j < c[1] + dim; ++j) {
			const float x2y2 = openvdb::math::Pow2(j - c[1]) + x2;
			for (k = c[2] - dim; k < c[2] + dim; ++k) {
				const float dist = openvdb::math::Sqrt(x2y2 + openvdb::math::Pow2(k - c[2])) - radius;
				auto val = ValueT(dist);
				if (val < inside || outside < val) continue;
				accessor.setValue(ijk, val);
			}
		}
	}

	openvdb::tools::signedFloodFill(grid->tree());

	return grid;
}

static openvdb::FloatGrid::Ptr sphere = makeSphere<openvdb::FloatGrid>(500, openvdb::Vec3f(0, 0, 0));


void EncodingDecodingTest(AllocationType type) {
	openvdb::FloatGrid::Ptr denPtr = sphere;
	denPtr->setName("density");
	denPtr->setGridClass(openvdb::GRID_FOG_VOLUME);

	openvdb::VectorGrid::Ptr velPtr = openvdb::VectorGrid::create();
	velPtr->setName("velocity");
	velPtr->setGridClass(openvdb::GRID_STAGGERED);

	openvdb::FloatGrid::Ptr domain = openvdb::FloatGrid::create();
	domain->topologyUnion(*denPtr);
	domain->topologyUnion(*velPtr);


	HNS::GridIndexedData indexed_data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &indexed_data);
	builder.setAllocType(type);

	builder.addGrid(denPtr, "density");
	builder.addGrid(velPtr, "velocity");

	builder.build();

	auto outDen = builder.writeIndexGrid<openvdb::FloatGrid>("density", domain->voxelSize()[0]);
	auto outVel = builder.writeIndexGrid<openvdb::VectorGrid>("velocity", domain->voxelSize()[0]);

	outDen->print();
	outVel->print(std::cout, 2);
}

void IndexGridModification(AllocationType type) {
	openvdb::FloatGrid::Ptr denPtr = openvdb::FloatGrid::create();
	denPtr->setName("density");
	denPtr->setGridClass(openvdb::GRID_FOG_VOLUME);

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
	builder.setAllocType(type);

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

void TrilinearSamplerTest(AllocationType type) {
	openvdb::FloatGrid::Ptr denPtr = openvdb::FloatGrid::create();
	denPtr->setName("density");
	denPtr->setGridClass(openvdb::GRID_FOG_VOLUME);

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
	builder.setAllocType(type);

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

void IndexSamplerTest(AllocationType type) {
	openvdb::FloatGrid::Ptr denPtr = openvdb::FloatGrid::create();
	denPtr->setName("density");
	denPtr->setGridClass(openvdb::GRID_FOG_VOLUME);

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
	builder.setAllocType(type);

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
	IndexSampler<float, 0> custom_index_f(idxSampler, densityPtr);
	IndexSampler<openvdb::Vec3f, 0> custom_index_v(idxSampler, velocityPtr);

	for (size_t i = 0; i < 15; ++i) {
		EXPECT_FLOAT_EQ(custom_index_f(nanovdb::Coord(i, 0, 0)), i);
		EXPECT_EQ(custom_index_v(nanovdb::Coord(0, i, 0)), openvdb::Vec3f(i, i, i));
	}
}


void IndexGridBuilderTest(AllocationType type) {
	openvdb::FloatGrid::Ptr gridPtr = openvdb::FloatGrid::create();
	gridPtr->setName("density");
	gridPtr->setGridClass(openvdb::GRID_FOG_VOLUME);

	/*
	 * -------------
	 * | 0 | 1 | 0 |
	 * | 1 | 1 | 1 |
	 * | 0 | 1 | 0 |
	 * -------------
	 *
	 #1#
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

	openvdb::VectorGrid::Ptr vecGrid = openvdb::VectorGrid::create();
	vecGrid->setName("velocity");
	gridPtr->setGridClass(openvdb::GRID_STAGGERED);

	auto vecGrid_acc = vecGrid->getAccessor();
	for (auto iter = gridPtr->cbeginValueOn(); iter; ++iter) {
		auto coord = iter.getCoord();

		// for each coord set the neighboring coord too
		const auto above = coord + openvdb::Coord(0, 1, 0);
		const auto below = coord + openvdb::Coord(0, -1, 0);
		const auto left = coord + openvdb::Coord(-1, 0, 0);
		const auto right = coord + openvdb::Coord(1, 0, 0);

		vecGrid_acc.setValue(coord, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		vecGrid_acc.setValue(above, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		vecGrid_acc.setValue(below, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		vecGrid_acc.setValue(left, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
		vecGrid_acc.setValue(right, openvdb::Vec3f(1.0f, 1.0f, 1.0f));
	}

	acc.setValue(center, 420.0f);
	vecGrid_acc.setValue(center, openvdb::Vec3f(420.0f, 420.0f, 420.0f));


	openvdb::FloatGrid::Ptr domain = openvdb::FloatGrid::create();
	domain->topologyUnion(*vecGrid);

	HNS::GridIndexedData indexed_data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &indexed_data);
	builder.setAllocType(type);

	builder.addGrid(gridPtr, "density");
	builder.addGrid(vecGrid, "velocity");

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
}

TEST(GridIndexedDataTest, AllocateCoordsAndAddBlocks) {
	HNS::GridIndexedData grid;

	constexpr size_t numElements = 5;
	bool success = grid.allocateCoords(numElements);
	EXPECT_TRUE(success);
	EXPECT_EQ(grid.size(), numElements);

	success = grid.addValueBlock<float>("density", numElements);
	EXPECT_TRUE(success);

	auto* densityBlock = grid.getValueBlock<float>("density");
	ASSERT_NE(densityBlock, nullptr);
	EXPECT_EQ(densityBlock->size(), numElements);
	auto* densityPtr = grid.pValues<float>("density");
	ASSERT_NE(densityPtr, nullptr);

	for (size_t i = 0; i < numElements; ++i) {
		densityPtr[i] = static_cast<float>(i) * 1.1f;
	}

	success = grid.addValueBlock<Vec3f>("velocity", numElements);
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
		EXPECT_FLOAT_EQ(densityPtr[i], static_cast<float>(i) * 1.1f);
		EXPECT_FLOAT_EQ(velocityPtr[i].x, static_cast<float>(i) + 0.1f);
		EXPECT_FLOAT_EQ(velocityPtr[i].y, static_cast<float>(i) + 0.2f);
		EXPECT_FLOAT_EQ(velocityPtr[i].z, static_cast<float>(i) + 0.3f);
	}
}

TEST(GridIndexedDataTest, ClearBlocks) {
	HNS::GridIndexedData grid;
	constexpr size_t numElements = 3;
	bool success = grid.allocateCoords(numElements);
	EXPECT_TRUE(success);

	success = grid.addValueBlock<float>("density", numElements);
	EXPECT_TRUE(success);

	auto* density = grid.pValues<float>("density");
	for (size_t i = 0; i < numElements; ++i) {
		density[i] = static_cast<float>(i);
	}

	grid.clearValues();
	EXPECT_EQ(grid.numValueBlocks(), static_cast<size_t>(0));
	EXPECT_EQ(grid.size(), numElements);

	grid.clear();
	EXPECT_EQ(grid.size(), static_cast<size_t>(0));
	EXPECT_EQ(grid.numValueBlocks(), static_cast<size_t>(0));
}

TEST(GridIndexedDataTest, MiniSampler) {
	openvdb::FloatGrid::Ptr gridPtr = openvdb::FloatGrid::create();
	gridPtr->setName("density");
	gridPtr->setGridClass(openvdb::GRID_FOG_VOLUME);
	gridPtr->getAccessor().setValue(openvdb::Coord(0, 0, 0), 1.0f);

	openvdb::FloatGrid::Ptr domain = openvdb::FloatGrid::create();
	domain->topologyUnion(*gridPtr);
	domain->getAccessor().setValue(openvdb::Coord(0, 1, 0), 1.0f);
	domain->getAccessor().setValue(openvdb::Coord(0, -1, 0), 1.0f);
	domain->getAccessor().setValue(openvdb::Coord(1, 0, 0), 1.0f);
	domain->getAccessor().setValue(openvdb::Coord(-1, 0, 0), 1.0f);
	domain->getAccessor().setValue(openvdb::Coord(0, 0, 1), 1.0f);
	domain->getAccessor().setValue(openvdb::Coord(0, 0, -1), 1.0f);


	HNS::GridIndexedData indexed_data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &indexed_data);
	builder.setAllocType(AllocationType::Standard);
	builder.addGrid(gridPtr, "density");
	builder.build();

	using SrcGridT = openvdb::FloatGrid;
	using DstBuildT = nanovdb::ValueOnIndex;
	using BufferT = nanovdb::HostBuffer;

	nanovdb::GridHandle<BufferT> idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*domain, 1u, false, false, 0);
	auto indexGrid = idxHandle.grid<DstBuildT>();

	nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> chanAccess(*indexGrid);
	IndexOffsetSampler<0> idxSampler(*indexGrid);
	IndexSampler<float, 0> samplerf(idxSampler, indexed_data.pValues<float>("density"));

	printf("Sampler Value : %f\n", samplerf(nanovdb::Coord(0, 0, 0)));
	printf("Channel Value : %f\n", chanAccess(0, 0, 0));
	printf("Offset Value : %llu\n", idxSampler.offset(nanovdb::Coord(0, 0, 0)));
	printf("Value in Array at Offset : %f\n", indexed_data.pValues<float>("density")[idxSampler.offset(nanovdb::Coord(0, 0, 0))]);
	EXPECT_FLOAT_EQ(samplerf(nanovdb::Coord(0, 0, 0)), 1.0f);
	EXPECT_FLOAT_EQ(samplerf(nanovdb::Coord(0, 1, 0)), 0.0f);

	for (int i = 0; i < indexed_data.size(); ++i) {
		if (indexed_data.pValues<float>("density")[i] == 1) {
			printf("Iter : %d\n", i);
			printf("Coord : %d %d %d\n", indexed_data.pCoords()[i].x(), indexed_data.pCoords()[i].y(), indexed_data.pCoords()[i].z());
		}
	}
}


TEST(GridIndexedDataTest, IndexGridBuilder_Standard) { IndexGridBuilderTest(AllocationType::Standard); }
TEST(GridIndexedDataTest, IndexSampler_Standard) { IndexSamplerTest(AllocationType::Standard); }
TEST(GridIndexedDataTest, TrilinearSampler_Standard) { TrilinearSamplerTest(AllocationType::Standard); }
TEST(GridIndexedDataTest, IndexGridModification_Standard) { IndexGridModification(AllocationType::Standard); }
TEST(GridIndexedDataTest, EncodingDecoding_Standard) { EncodingDecodingTest(AllocationType::Standard); }

TEST(GridIndexedDataTest, IndexGridBuilder_Aligned) { IndexGridBuilderTest(AllocationType::Aligned); }
TEST(GridIndexedDataTest, IndexSampler_Aligned) { IndexSamplerTest(AllocationType::Aligned); }
TEST(GridIndexedDataTest, TrilinearSampler_Aligned) { TrilinearSamplerTest(AllocationType::Aligned); }
TEST(GridIndexedDataTest, IndexGridModification_Aligned) { IndexGridModification(AllocationType::Aligned); }
TEST(GridIndexedDataTest, EncodingDecoding_Aligned) { EncodingDecodingTest(AllocationType::Aligned); }

TEST(GridIndexedDataTest, IndexGridBuilder_Pinned) { IndexGridBuilderTest(AllocationType::CudaPinned); }
TEST(GridIndexedDataTest, IndexSampler_Pinned) { IndexSamplerTest(AllocationType::CudaPinned); };
TEST(GridIndexedDataTest, TrilinearSampler_Pinned) { TrilinearSamplerTest(AllocationType::CudaPinned); }
TEST(GridIndexedDataTest, IndexGridModification_Pinned) { IndexGridModification(AllocationType::CudaPinned); }
TEST(GridIndexedDataTest, EncodingDecoding_Pinned) { EncodingDecodingTest(AllocationType::CudaPinned); }
