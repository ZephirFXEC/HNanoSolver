#include <nanovdb/util/GridBuilder.h>

#include <nanovdb/util/cuda/CudaPointsToGrid.cuh>

#include "../Utils/GridData.hpp"
#include "utils.cuh"

/*
 * InputCoordT: The type of the input coordinates (e.g., openvdb::Coord)
 * InputValueT: The type of the input values (e.g., float or openvdb::Vec3f)
 *
 * OutputCoordT: The type of the output coordinates (e.g., nanovdb::Coord)
 * OutputValueT: The type of the output values (e.g., float or nanovdb::Vec3f)
 *
 * N
 */
template <typename InputCoordT, typename InputValueT, typename OutputCoordT, typename OutputValueT,
          typename NanoGridType, typename NanoOpT>
void pointToGridTemplate(const GridData<InputCoordT, InputValueT>& in_data, const float voxelSize,
                         GridData<OutputCoordT, OutputValueT>& out_data) {
	const size_t npoints = in_data.size;

	// Allocate and copy coordinates to the device
	OutputCoordT* d_coords = nullptr;
	cudaCheck(cudaMalloc(&d_coords, npoints * sizeof(OutputCoordT)));
	cudaCheck(cudaMemcpyAsync(d_coords, (OutputCoordT*)in_data.pCoords, npoints * sizeof(OutputCoordT),
	                          cudaMemcpyHostToDevice));

	// Generate a NanoVDB grid that contains the list of voxels on the device
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle =
	    nanovdb::cudaVoxelsToGrid<OutputValueT>(d_coords, npoints, voxelSize);
	NanoGridType* d_grid = handle.deviceGrid<OutputValueT>();

	// Allocate and copy values to the device
	OutputValueT* d_values;
	cudaCheck(cudaMalloc(&d_values, npoints * sizeof(OutputValueT)));
	cudaCheck(cudaMemcpyAsync(d_values, reinterpret_cast<const OutputValueT*>(in_data.pValues),
	                          npoints * sizeof(OutputValueT), cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());

	// Launch a device kernel to set the voxel values
	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	lambdaKernel<<<numBlocks, numThreads>>>(npoints, [=] __device__(const size_t tid) {
		const OutputCoordT& ijk = d_coords[tid];
		d_grid->tree().template set<NanoOpT>(ijk, d_values[tid]);
	});
	cudaCheckError();

	cudaCheck(cudaDeviceSynchronize());

	// Copy results back to the host
	cudaCheck(cudaMemcpyAsync(out_data.pValues, d_values, sizeof(OutputValueT) * npoints, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpyAsync(out_data.pCoords, d_coords, sizeof(OutputCoordT) * npoints, cudaMemcpyDeviceToHost));

	// Free device memory
	cudaCheck(cudaFree(d_coords));
	cudaCheck(cudaFree(d_values));
}

template <typename InputCoordT, typename InputValueT, typename OutputCoordT, typename OutputValueT,
          typename NanoGridType, typename NanoOpT>
void pointToGridTemplateToDevice(const GridData<InputCoordT, InputValueT>& in_data, const float voxelSize,
                                 nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle) {
	const size_t npoints = in_data.size;

	// Allocate and copy coordinates to the device
	OutputCoordT* d_coords = nullptr;
	cudaCheck(cudaMalloc(&d_coords, npoints * sizeof(OutputCoordT)));
	cudaCheck(cudaMemcpyAsync(d_coords, (OutputCoordT*)in_data.pCoords, npoints * sizeof(OutputCoordT),
	                          cudaMemcpyHostToDevice));

	// Generate a NanoVDB grid that contains the list of voxels on the device
	handle = nanovdb::cudaVoxelsToGrid<OutputValueT>(d_coords, npoints, voxelSize);
	NanoGridType* d_grid = handle.deviceGrid<OutputValueT>();

	// Allocate and copy values to the device
	OutputValueT* d_values;
	cudaCheck(cudaMalloc(&d_values, npoints * sizeof(OutputValueT)));
	cudaCheck(cudaMemcpyAsync(d_values, reinterpret_cast<const OutputValueT*>(in_data.pValues),
	                          npoints * sizeof(OutputValueT), cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());

	// Launch a device kernel to set the voxel values
	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	lambdaKernel<<<numBlocks, numThreads>>>(npoints, [=] __device__(const size_t tid) {
		const OutputCoordT& ijk = d_coords[tid];
		d_grid->tree().template set<NanoOpT>(ijk, d_values[tid]);
	});
	cudaCheckError();

	// Free device memory
	cudaCheck(cudaFree(d_coords));
	cudaCheck(cudaFree(d_values));
}

extern "C" void pointToGridFloat(const OpenFloatGrid& in_data, const float voxelSize, NanoFloatGrid& out_data) {
	pointToGridTemplate<openvdb::Coord, float, nanovdb::Coord, float, nanovdb::FloatGrid, nanovdb::SetVoxel<float>>(
	    in_data, voxelSize, out_data);
}

extern "C" void pointToGridVector(const OpenVectorGrid& in_data, const float voxelSize, NanoVectorGrid& out_data) {
	pointToGridTemplate<openvdb::Coord, openvdb::Vec3f, nanovdb::Coord, nanovdb::Vec3f, nanovdb::Vec3fGrid,
	                    nanovdb::SetVoxel<nanovdb::Vec3f>>(in_data, voxelSize, out_data);
}

extern "C" void pointToGridFloatToDevice(const OpenFloatGrid& in_data, const float voxelSize,
                                         nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle) {
	pointToGridTemplateToDevice<openvdb::Coord, float, nanovdb::Coord, float, nanovdb::FloatGrid,
	                            nanovdb::SetVoxel<float>>(in_data, voxelSize, handle);
}

extern "C" void pointToGridVectorToDevice(const OpenVectorGrid& in_data, const float voxelSize,
                                          nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle) {
	pointToGridTemplateToDevice<openvdb::Coord, openvdb::Vec3f, nanovdb::Coord, nanovdb::Vec3f, nanovdb::Vec3fGrid,
	                            nanovdb::SetVoxel<nanovdb::Vec3f>>(in_data, voxelSize, handle);
}
