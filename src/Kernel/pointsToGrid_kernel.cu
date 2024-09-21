#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/cuda/CudaPointsToGrid.cuh>

#include "utils.cuh"
#include "../Utils/GridData.hpp"


extern "C" void pointToGrid(const std::vector<nanovdb::Coord>& coords, const std::vector<float>& values, const float voxelSize, NanoFloatGrid& out_data) {


	const size_t npoints = coords.size();
	nanovdb::Coord* d_coords = nullptr;
	cudaCheck(cudaMalloc(&d_coords, npoints * sizeof(nanovdb::Coord)));
	cudaCheck(cudaMemcpyAsync(d_coords, coords.data(), npoints * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

	// Generate a NanoVDB grid that contains the list of voxels on the device
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle = nanovdb::cudaVoxelsToGrid<float>(d_coords, npoints, voxelSize);
	nanovdb::FloatGrid* d_grid = handle.deviceGrid<float>();

	// Define a list of values and copy them to the device
	float *d_values;
	cudaCheck(cudaMalloc(&d_values, npoints * sizeof(float)));
	cudaCheck(cudaMemcpyAsync(d_values, values.data(), npoints * sizeof(float), cudaMemcpyHostToDevice));

	// Synchronize to ensure all data is copied before launching the kernel
	cudaCheck(cudaDeviceSynchronize());

	// Launch a device kernel that sets the values of voxels define above and prints them
	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);
	using OpT = nanovdb::SetVoxel<float>;

	lambdaKernel<<<numBlocks, numThreads>>>(npoints, [=] __device__(const size_t tid) {
		const nanovdb::Coord &ijk = d_coords[tid];
		d_grid->tree().set<OpT>(ijk, d_values[tid]);// normally one should use a ValueAccessor
	}); cudaCheckError();


	cudaCheck(cudaMalloc(&out_data.pValues, sizeof(float) * npoints));
	cudaCheck(cudaMalloc(&out_data.pCoords, sizeof(nanovdb::Coord) * npoints));
	cudaCheck(cudaMemcpyAsync(out_data.pValues, d_values, sizeof(float), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpyAsync(out_data.pCoords, d_grid, sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost));

	// free arrays allocated on the device
	cudaCheck(cudaFree(d_coords));
	cudaCheck(cudaFree(d_values));
}