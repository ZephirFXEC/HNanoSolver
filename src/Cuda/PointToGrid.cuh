#pragma once

#include "../Utils/GridData.hpp"
#include "HNanoGrid/HNanoGrid.cuh"
#include "utils.cuh"

template <typename ValueOutT, bool HasTemp>
void pointToTopologyToDevice(CudaResources<ValueOutT, HasTemp>& resources, openvdb::Coord* h_coords, const size_t npoints,
                             const float voxelSize, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {

	resources.LoadPointCoord(h_coords, npoints, stream);

	handle = nanovdb::cudaVoxelsToGrid<ValueOutT>(resources.d_coords, npoints, voxelSize);
}

template <typename ValueOutT, typename InType, bool HasTemp>
void fillTopology(CudaResources<ValueOutT, HasTemp>& resources, InType* h_values, const size_t npoints,
                  nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	using TreeT = std::conditional_t<std::is_same_v<float, ValueOutT>, nanovdb::FloatTree, nanovdb::Vec3fTree>;

	nanovdb::Grid<TreeT>* grid = handle.deviceGrid<ValueOutT>();

	resources.template LoadPointValue<InType>(h_values, npoints, stream);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = (npoints + numThreads - 1) / numThreads;

	// Launch kernel to set grid values
	set_grid_values<ValueOutT, TreeT, HasTemp><<<numBlocks, numThreads, 0, stream>>>(resources, npoints, grid);
	cudaCheck(cudaGetLastError());
}


template <typename ValueInT, typename ValueOutT>
void pointToGridTemplate(HNS::OpenGrid<ValueInT>& in_data, const float voxelSize, HNS::NanoGrid<ValueOutT>& out_data,
                         const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<ValueOutT, true> resources(npoints, stream);
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle;
	pointToTopologyToDevice<ValueOutT, true>(resources, in_data.pCoords(), npoints, voxelSize, handle, stream);
	fillTopology<ValueOutT, ValueInT, true>(resources, in_data.pValues(), npoints, handle, stream);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	using TreeT = std::conditional_t<std::is_same_v<float, ValueOutT>, nanovdb::FloatTree, nanovdb::Vec3fTree>;
	nanovdb::Grid<TreeT>* d_grid = handle.deviceGrid<ValueOutT>();
	get_grid_values<ValueOutT><<<numBlocks, numThreads, 0, stream>>>(resources, npoints, d_grid);

	out_data.allocateCudaPinned(npoints);

	resources.UnloadPointData(out_data, stream);

	resources.cleanup(stream);
}


template <typename ValueInT, typename ValueOutT>
void pointToGridTemplateToDevice(HNS::OpenGrid<ValueInT>& in_data, const float voxelSize,
                                 nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<ValueOutT, false> resources(npoints, stream);

	pointToTopologyToDevice<ValueOutT, false>(resources, in_data.pCoords(), npoints, voxelSize, handle, stream);

	fillTopology<ValueOutT, ValueInT, false>(resources, in_data.pValues(), npoints, handle, stream);

	resources.cleanup(stream);
}