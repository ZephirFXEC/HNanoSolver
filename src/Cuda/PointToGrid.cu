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

	CudaResources<ValueOutT> resources(npoints, stream);
	resources.template LoadPointData<ValueInT>(in_data, stream);

	cudaCheck(cudaStreamWaitEvent(stream, resources.CoordBeenCopied, 0));
	auto handle = nanovdb::cudaVoxelsToGrid<ValueOutT>(resources.d_coords, npoints, voxelSize);
	nanovdb::NanoGrid<ValueOutT>* d_grid = handle.template deviceGrid<ValueOutT>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	cudaCheck(cudaStreamWaitEvent(stream, resources.ValueBeenCopied, 0));
	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const auto accessor = d_grid->tree().getAccessor();
		accessor.template set<nanovdb::SetVoxel<ValueOutT>>(resources.d_coords[tid], resources.d_values[tid]);
	});

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const auto accessor = d_grid->tree().getAccessor();
		resources.d_temp_values[tid] = accessor.getValue(resources.d_coords[tid]);
	});

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


template <typename InputCoordT, typename InputValueT, typename OutputCoordT, typename OutputValueT, typename NanoGridType, typename NanoOpT>
void pointToGridTemplateToDevice(HNS::OpenGrid<InputValueT>& in_data, const float voxelSize,
                                 nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	cudaCheck(cudaHostRegister(in_data.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(in_data.pValues(), npoints * sizeof(InputValueT), cudaHostRegisterDefault));

	CudaResources<OutputValueT> resources(npoints, stream);
	resources.template LoadPointData<InputValueT>(in_data, stream);

	cudaStreamWaitEvent(stream, resources.CoordBeenCopied, 0);
	handle = nanovdb::cudaVoxelsToGrid<OutputValueT>(resources.d_coords, npoints, voxelSize);
	nanovdb::NanoGrid<OutputValueT>* d_grid = handle.deviceGrid<OutputValueT>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	cudaCheck(cudaStreamWaitEvent(stream, resources.ValueBeenCopied, 0));
	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const OutputCoordT& ijk = resources.d_coords[tid];
		d_grid->tree().template set<NanoOpT>(ijk, resources.d_values[tid]);
	});
	cudaCheckError();

	cudaCheck(cudaHostUnregister(in_data.pCoords()));
	cudaCheck(cudaHostUnregister(in_data.pValues()));

	resources.clear(stream);
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
	pointToGridTemplateToDevice<openvdb::Coord, float, nanovdb::Coord, float, nanovdb::FloatGrid, nanovdb::SetVoxel<float>>(
	    in_data, voxelSize, handle, stream);
}

extern "C" void pointToGridVectorToDevice(HNS::OpenVectorGrid& in_data, const float voxelSize,
                                          nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	pointToGridTemplateToDevice<openvdb::Coord, openvdb::Vec3f, nanovdb::Coord, nanovdb::Vec3f, nanovdb::Vec3fGrid,
	                            nanovdb::SetVoxel<nanovdb::Vec3f>>(in_data, voxelSize, handle, stream);
}
