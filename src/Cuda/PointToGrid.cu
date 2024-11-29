#include "../Utils/GridData.hpp"
#include "HNanoGrid/HNanoGrid.cuh"
#include "utils.cuh"

/*
 * InputCoordT: The type of the input coordinates (e.g., openvdb::Coord)
 * InputValueT: The type of the input values (e.g., float or openvdb::Vec3f)
 *
 * OutputCoordT: The type of the output coordinates (e.g., nanovdb::Coord)
 * OutputValueT: The type of the output values (e.g., float or nanovdb::Vec3f)
 *
 * NanoGridType: The type of the NanoVDB grid (e.g., nanovdb::FloatGrid or nanovdb::Vec3fGrid)
 * NanoOpT: The type of the NanoVDB operation (e.g., nanovdb::SetVoxel<float> or nanovdb::SetVoxel<nanovdb::Vec3f>)
 */

template <typename ValueInT, typename ValueOutT>
void pointToGridTemplate(HNS::OpenGrid<ValueInT>& in_data, const float voxelSize, HNS::NanoGrid<ValueOutT>& out_data,
                         const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	cudaCheck(cudaHostRegister(in_data.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(in_data.pValues(), npoints * sizeof(ValueInT), cudaHostRegisterDefault));

	CudaResources<ValueOutT, true> resources(npoints, stream);
	resources.template LoadPointData<ValueInT>(in_data, stream);

	cudaCheck(cudaStreamSynchronize(stream));

	auto handle = nanovdb::cudaVoxelsToGrid<ValueOutT>(resources.d_coords, npoints, voxelSize);
	nanovdb::NanoGrid<ValueOutT>* d_grid = handle.template deviceGrid<ValueOutT>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	set_grid_values<ValueOutT><<<numBlocks, numThreads, 0, stream>>>(resources, npoints, d_grid);

	get_grid_values<ValueOutT><<<numBlocks, numThreads, 0, stream>>>(resources, npoints, d_grid);

	out_data.allocateStandard(npoints);

	cudaCheck(cudaHostRegister(out_data.pCoords(), npoints * sizeof(nanovdb::Coord), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(out_data.pValues(), npoints * sizeof(ValueOutT), cudaHostRegisterDefault));

	resources.UnloadPointData(out_data, stream);

	cudaCheck(cudaHostUnregister(in_data.pCoords()));
	cudaCheck(cudaHostUnregister(in_data.pValues()));
	cudaCheck(cudaHostUnregister(out_data.pCoords()));
	cudaCheck(cudaHostUnregister(out_data.pValues()));

	resources.cleanup(stream);
}


template <typename ValueInT, typename ValueOutT>
void pointToGridTemplateToDevice(HNS::OpenGrid<ValueInT>& in_data, const float voxelSize,
                                 nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	cudaCheck(cudaHostRegister(in_data.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(in_data.pValues(), npoints * sizeof(ValueInT), cudaHostRegisterDefault));

	CudaResources<ValueOutT, true> resources(npoints, stream);
	resources.template LoadPointData<ValueInT>(in_data, stream);

	cudaCheck(cudaStreamSynchronize(stream));

	handle = nanovdb::cudaVoxelsToGrid<ValueOutT>(resources.d_coords, npoints, voxelSize);
	nanovdb::NanoGrid<ValueOutT>* d_grid = handle.deviceGrid<ValueOutT>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	set_grid_values<ValueOutT><<<numBlocks, numThreads, 0, stream>>>(resources, npoints, d_grid);

	cudaCheck(cudaHostUnregister(in_data.pCoords()));
	cudaCheck(cudaHostUnregister(in_data.pValues()));

	resources.cleanup(stream);
}

extern "C" void pointToGridFloat(HNS::OpenFloatGrid& in_data, const float voxelSize, HNS::NanoFloatGrid& out_data,
                                 const cudaStream_t& stream) {
	pointToGridTemplate<float, float>(in_data, voxelSize, out_data, stream);
}

extern "C" void pointToGridVector(HNS::OpenVectorGrid& in_data, const float voxelSize, HNS::NanoVectorGrid& out_data,
                                  const cudaStream_t& stream) {
	pointToGridTemplate<openvdb::Vec3f, nanovdb::Vec3f>(in_data, voxelSize, out_data, stream);
}

extern "C" void pointToGridFloatToDevice(HNS::OpenFloatGrid& in_data, const float voxelSize,
                                         nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	pointToGridTemplateToDevice<float, float>(
	    in_data, voxelSize, handle, stream);
}

extern "C" void pointToGridVectorToDevice(HNS::OpenVectorGrid& in_data, const float voxelSize,
                                          nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	pointToGridTemplateToDevice<openvdb::Vec3f, nanovdb::Vec3f>(in_data, voxelSize, handle, stream);
}
