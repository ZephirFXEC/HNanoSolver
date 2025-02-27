#include <openvdb/Types.h>

#include <cstdio>  // for printf

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

__global__ void sampler_gpu(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid, const nanovdb::Coord* coords,
                            const nanovdb::Vec3f* velocity, const float* density) {
	const IndexOffsetSampler<0> idxSampler(*gpuGrid);

	const IndexSampler<float, 1> sampler_f(idxSampler, density);
	const IndexSampler<nanovdb::Vec3f, 1> sampler_v(idxSampler, velocity);


	const nanovdb::Vec3f center = coords[0].asVec3s();

	const float density000 = sampler_f(center);
	const float density100 = sampler_f(center + nanovdb::Vec3f(0.5, 0, 0));
	const float density010 = sampler_f(center + nanovdb::Vec3f(0, 0.5, 0));
	const float density001 = sampler_f(center + nanovdb::Vec3f(0, 0, 0.5));

	printf("Sampling Density Grid %f%f%f: \n", center[0], center[1], center[2]);
	printf("Density at : 0,0,0 =  %f\n", density000);
	printf("Density at : 0.5,0,0 =  %f\n", density100);
	printf("Density at : 0,0.5,0 =  %f\n", density010);
	printf("Density at : 0,0,0.5 =  %f\n", density001);

	const nanovdb::Vec3f vel000 = sampler_v(center);
	const nanovdb::Vec3f vel100 = sampler_v(center + nanovdb::Vec3f(0.5, 0, 0));
	const nanovdb::Vec3f vel010 = sampler_v(center + nanovdb::Vec3f(0, 0.5, 0));
	const nanovdb::Vec3f vel001 = sampler_v(center + nanovdb::Vec3f(0, 0, 0.5));

	printf("Sampling Velocity Grid %f%f%f: \n", center[0], center[1], center[2]);
	printf("Velocity at : 0,0,0 =  {%f, %f, %f}\n", vel000[0], vel000[1], vel000[2]);
	printf("Velocity at : 0.5,0,0 =  {%f, %f, %f}\n", vel100[0], vel100[1], vel100[2]);
	printf("Velocity at : 0,0.5,0 =  {%f, %f, %f}\n", vel010[0], vel010[1], vel010[2]);
	printf("Velocity at : 0,0,0.5 =  {%f, %f, %f}\n", vel001[0], vel001[1], vel001[2]);
}

__global__ void advect_idx(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
                           const nanovdb::Coord* coords,        // Device array of size totalVoxels
                           const nanovdb::Vec3f* velocityData,  // Device array of size totalVoxels
                           const float* densityData,            // Device array of size totalVoxels
                           float* outDensity,                   // Output array of size totalVoxels
                           const size_t totalVoxels, const float dt, const float voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;  // Early out-of-bounds check

	// Construct sampler objects AFTER ensuring the thread is within bounds.
	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto densitySampler = IndexSampler<float, 1>(idxSampler, densityData);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);

	// Get the active coordinate for this voxel.
	const nanovdb::Coord coord = coords[idx];

	if (idx == 16) printf("Coord and idx : [%llu]: {%d %d %d}\n", idx, coord[0], coord[1], coord[2]);

	// Fetch the velocity at the given coordinate.
	const nanovdb::Vec3f velocity = velocitySampler(coord);

	if (idx == 16)
		printf("Velocity idx : [%llu] at %d %d %d = {%f %f %f}\n", idx, coord[0], coord[1], coord[2], velocity[0], velocity[1],
		       velocity[2]);

	// Compute the displaced position.
	const nanovdb::Vec3f displacedPos = coord.asVec3s() - velocity * dt / voxelSize;

	if (idx == 16) printf("Displaced Position idx : [%llu] at {%f %f %f}\n", idx, displacedPos[0], displacedPos[1], displacedPos[2]);

	// Sample the density at the displaced position.
	outDensity[idx] = densitySampler(displacedPos);

	if (idx == 16)
		printf("Density idx : [%llu] at {%f %f %f} = %f\n", idx, displacedPos[0], displacedPos[1], displacedPos[2], outDensity[idx]);
}

extern "C" void gpu_index_grid(HNS::GridIndexedData& data, const float voxelSize, const cudaStream_t& stream) {
	const auto* h_coords = data.pCoords();
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_coords, data.size() * sizeof(nanovdb::Coord));
	cudaMemcpyAsync(d_coords, h_coords, data.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);


	cudaGetLastError();
}


// This is called by the client code on the host
extern "C" void launch_kernels(HNS::GridIndexedData& data,
                               const float dt,         // time step for advection
                               const float voxelSize,  // voxel size in world units
                               cudaStream_t stream) {
	const size_t totalVoxels = data.size();

	const nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>("velocity"));
	const auto* density = data.pValues<float>("density");
	const auto* coords = data.pCoords();

	// Allocate device memory for velocity.
	nanovdb::Vec3f* d_velocity = nullptr;
	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);

	// Allocate device memory for density.
	float* d_density = nullptr;
	cudaMalloc(&d_density, totalVoxels * sizeof(float));
	cudaMemcpyAsync(d_density, density, totalVoxels * sizeof(float), cudaMemcpyHostToDevice, stream);

	// Allocate device memory for voxel coordinates.
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMemcpyAsync(d_coords, coords, totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);


	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle =
	    nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	// Launch the advection kernel.
	sampler_gpu<<<1, 1, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_density);

	// Wait for all asynchronous operations on this stream to finish.
	cudaStreamSynchronize(stream);

	// Free the allocated device memory.
	cudaFree(d_velocity);
	cudaFree(d_density);
	cudaFree(d_coords);
}