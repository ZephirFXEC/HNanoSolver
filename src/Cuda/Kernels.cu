// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/NanoVDB.h>  // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <openvdb/openvdb.h>

#include <cstdio>  // for printf
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include "utils.cuh"

#include "../Utils/Stencils.hpp"

__global__ void sampler_gpu(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid, const nanovdb::Vec3f* velocity, const float* density) {

	const nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> acc(*gpuGrid);


	const IndexOffsetSampler<0> idxSampler(*gpuGrid);

	const IndexSampler<float, 1> sampler_f(idxSampler, density);
	const IndexSampler<nanovdb::Vec3f, 1> sampler_v(idxSampler, velocity);

	if (!acc) {
		printf("Error: ChannelAccessor has no valid channel pointer! (null)\n");
		return;
	}

	const nanovdb::Vec3f center(0, 0, 0);

	const float density000 = sampler_f(center);
	const float density100 = sampler_f(center + nanovdb::Vec3f(0.5, 0, 0));
	const float density010 = sampler_f(center + nanovdb::Vec3f(0, 0.5, 0));
	const float density001 = sampler_f(center + nanovdb::Vec3f(0, 0, 0.5));

	printf("Sampling Density Grid : \n");
	printf("Density at : 0,0,0 =  %f\n", density000);
	printf("Density at : 0.5,0,0 =  %f\n", density100);
	printf("Density at : 0,0.5,0 =  %f\n", density010);
	printf("Density at : 0,0,0.5 =  %f\n", density001);

	const nanovdb::Vec3f vel000 = sampler_v(center);
	const nanovdb::Vec3f vel100 = sampler_v(center + nanovdb::Vec3f(0.5, 0, 0));
	const nanovdb::Vec3f vel010 = sampler_v(center + nanovdb::Vec3f(0, 0.5, 0));
	const nanovdb::Vec3f vel001 = sampler_v(center + nanovdb::Vec3f(0, 0, 0.5));

	printf("Sampling Velocity Grid : \n");
	printf("Velocity at : 0,0,0 =  {%f, %f, %f}\n", vel000[0], vel000[1], vel000[2]);
	printf("Velocity at : 0.5,0,0 =  {%f, %f, %f}\n", vel100[0], vel100[1], vel100[2]);
	printf("Velocity at : 0,0.5,0 =  {%f, %f, %f}\n", vel010[0], vel010[1], vel010[2]);
	printf("Velocity at : 0,0,0.5 =  {%f, %f, %f}\n", vel001[0], vel001[1], vel001[2]);

}


// This is called by the client code on the host
extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid, void* velocity, void* density, size_t size, const
                               cudaStream_t stream) {

	nanovdb::Vec3f* vel = nullptr;
	cudaMalloc(&vel, size * sizeof(openvdb::Vec3f));
	cudaMemcpy(vel, velocity, size * sizeof(openvdb::Vec3f), cudaMemcpyHostToDevice);


	float* den = nullptr;
	cudaMalloc(&den, size * sizeof(float));
	cudaMemcpy(den, density, size * sizeof(float), cudaMemcpyHostToDevice);

	sampler_gpu<<<1, 1, 0, stream>>>(gpuGrid, vel, den);
}