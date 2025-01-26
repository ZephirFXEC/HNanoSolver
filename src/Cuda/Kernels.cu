// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/NanoVDB.h>  // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <openvdb/openvdb.h>

#include <cstdio>  // for printf
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/cuda/IndexToGrid.cuh>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include "../Utils/OpenToNano.hpp"
#include "utils.cuh"

__global__ void gpu_kernel(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid, const nanovdb::Vec3f* data) {

	const nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> acc(*gpuGrid);

	if (!acc) {
		printf("Error: ChannelAccessor has no valid channel pointer! (null)\n");
		return;
	}

	const uint32_t idx000 = acc.idx(0, 0, 0);
	const uint32_t idx100 = acc.idx(1, 0, 0);
	const uint32_t idx010 = acc.idx(0, 1, 0);
	const uint32_t idx001 = acc.idx(0, 0, 1);

	printf("Idx at Coord 0,0,0 : %u\n", idx000);
	printf("Idx at Coord 1,0,0 : %u\n", idx100);
	printf("Idx at Coord 0,1,0 : %u\n", idx010);
	printf("Idx at Coord 0,0,1 : %u\n", idx001);

	printf("Sampling Density Grid : \n");
	printf("Density at : %u =  %f\n", idx000, acc(0, 0, 0));
	printf("Density at : %u =  %f\n", idx100, acc(1, 0, 0));
	printf("Density at : %u =  %f\n", idx010, acc(0, 1, 0));
	printf("Density at : %u =  %f\n", idx001, acc(0, 0, 1));

	const nanovdb::Vec3f vel000 = data[idx000];
	const nanovdb::Vec3f vel100 = data[idx100];
	const nanovdb::Vec3f vel010 = data[idx010];
	const nanovdb::Vec3f vel001 = data[idx001];

	printf("Sampling Velocity Grid : \n");
	printf("Velocity at : %u =  {%f, %f, %f}\n", idx000, vel000[0], vel000[1], vel000[2]);
	printf("Velocity at : %u =  {%f, %f, %f}\n", idx100, vel100[0], vel100[1], vel100[2]);
	printf("Velocity at : %u =  {%f, %f, %f}\n", idx010, vel010[0], vel010[1], vel010[2]);
	printf("Velocity at : %u =  {%f, %f, %f}\n", idx001, vel001[0], vel001[1], vel001[2]);
}



// This is called by the client code on the host
extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid, void* data, size_t size, const
                               cudaStream_t stream) {

	nanovdb::Vec3f* vel = nullptr;
	cudaMalloc(&vel, size * sizeof(openvdb::Vec3f));
	cudaMemcpy(vel, data, size * sizeof(openvdb::Vec3f), cudaMemcpyHostToDevice);

	gpu_kernel<<<1, 1, 0, stream>>>(gpuGrid, vel); //res.d_values);  // Launch the device kernel asynchronously
}


extern "C" void openToNanoIndex(const openvdb::FloatGrid::Ptr& in_grid, const cudaStream_t stream) {
	const size_t voxelCount = in_grid->activeVoxelCount();
	const openvdb::Vec3d voxelSize = in_grid->voxelSize();

	// extract coords from openvdb grid
	std::vector<nanovdb::Vec3f> coords;
	coords.reserve(voxelCount);
	for (auto iter = in_grid->beginValueOn(); iter; ++iter) {
		const openvdb::Coord& coord = iter.getCoord();
		coords.emplace_back(coord.x(), coord.y(), coord.z());
	}

	nanovdb::Vec3f* coord_d = nullptr;

	cudaMalloc(&coord_d, voxelCount * sizeof(nanovdb::Vec3f));
	cudaMemcpy(coord_d, coords.data(), voxelCount * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice);

	const nanovdb::cuda::DeviceBuffer buffer(voxelCount * sizeof(float), false, stream);
	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle =
	    nanovdb::tools::cuda::voxelsToGrid<float>(coord_d, voxelCount, voxelSize[0], buffer, stream);

	const auto* gpuGrid = handle.deviceGrid<nanovdb::PointGrid>();
}