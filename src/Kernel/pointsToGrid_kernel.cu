#include <nanovdb/util/GridBuilder.h>

#include <nanovdb/util/cuda/CudaPointsToGrid.cuh>
#include <nanovdb/util/cuda/CudaIndexToGrid.cuh>

#include "utils.cuh"


struct Grid {
	std::vector<nanovdb::Coord> coords{};
	std::vector<float> values{};
	float voxelSize = 0.5f;
};

extern "C" void pointToGrid(const Grid& gridData, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& out_handle) {

	const size_t npoints = gridData.coords.size();
	nanovdb::Coord* d_coords = nullptr;
	cudaCheck(cudaMalloc(&d_coords, npoints * sizeof(nanovdb::Coord)));
	cudaCheck(cudaMemcpy(d_coords, gridData.coords.data(), npoints * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

	// Generate a NanoVDB grid that contains the list of voxels on the device
	out_handle = nanovdb::cudaVoxelsToGrid<float>(d_coords, npoints, gridData.voxelSize);
	auto *d_grid = out_handle.deviceGrid<float>();

	// Define a list of values and copy them to the device
	float *d_values;
	cudaCheck(cudaMalloc(&d_values, npoints * sizeof(float)));
	cudaCheck(cudaMemcpy(d_values, gridData.values.data(), npoints * sizeof(float), cudaMemcpyHostToDevice));// values CPU -> GPU

	// Launch a device kernel that sets the values of voxels define above and prints them
	constexpr unsigned int numThreads = 128;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);
	lambdaKernel<<<numBlocks, numThreads>>>(npoints, [=] __device__(const size_t tid) {
		using OpT = nanovdb::SetVoxel<float>;// defines type of random-access operation (set value)
		const nanovdb::Coord &ijk = d_coords[tid];
		d_grid->tree().set<OpT>(ijk, d_values[tid]);// normally one should use a ValueAccessor
	}); cudaCheckError();

	// free arrays allocated on the device
	cudaCheck(cudaFree(d_coords));
	cudaCheck(cudaFree(d_values));
}