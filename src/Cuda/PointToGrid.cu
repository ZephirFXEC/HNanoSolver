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
void pointToGridTemplate(const HNS::OpenGrid<ValueInT>& in_data, const float voxelSize,
                         HNS::NanoGrid<ValueOutT>& out_data, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<ValueOutT> resources(npoints, stream);
	HostMemoryManager<ValueInT, ValueOutT> memory_manager(in_data, out_data);
	LoadPointData<ValueInT, ValueOutT>(resources, in_data, npoints, stream);

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle =
	    nanovdb::cudaVoxelsToGrid<ValueOutT>(resources.d_coords, npoints, voxelSize);
	nanovdb::NanoGrid<ValueOutT>* d_grid = handle.deviceGrid<ValueOutT>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	// Set Grid
	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const auto accessor = d_grid->tree().getAccessor();
		accessor.template set<nanovdb::SetVoxel<ValueOutT>>(resources.d_coords[tid], resources.d_values[tid]);
	});

	// Get values from grid
	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		resources.d_temp_values[tid] = resources.d_values[tid];
	});

	UnloadPointData<ValueOutT>(resources, out_data, npoints, stream);

	resources.clear(stream);
}

template <typename InputCoordT, typename InputValueT, typename OutputCoordT, typename OutputValueT,
          typename NanoGridType, typename NanoOpT>
void pointToGridTemplateToDevice(const HNS::OpenGrid<InputValueT>& in_data, const float voxelSize,
                                 nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<OutputValueT> resources(npoints, stream);
	HostMemoryManager<InputValueT, OutputValueT> memory_manager(in_data, HNS::NanoGrid<OutputValueT>());
	LoadPointData<InputValueT, OutputValueT>(resources, in_data, npoints, stream);

	handle = nanovdb::cudaVoxelsToGrid<OutputValueT>(resources.d_coords, npoints, voxelSize);
	NanoGridType* d_grid = handle.deviceGrid<OutputValueT>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const OutputCoordT& ijk = resources.d_coords[tid];
		d_grid->tree().template set<NanoOpT>(ijk, resources.d_values[tid]);
	});
	cudaCheckError();

	resources.clear(stream);
}

extern "C" void pointToGridFloat(const HNS::OpenFloatGrid& in_data, const float voxelSize, HNS::NanoFloatGrid& out_data,
                                 const cudaStream_t& stream) {
	pointToGridTemplate<float, float>(in_data, voxelSize, out_data, stream);
}

extern "C" void pointToGridVector(const HNS::OpenVectorGrid& in_data, const float voxelSize,
                                  HNS::NanoVectorGrid& out_data, const cudaStream_t& stream) {
	pointToGridTemplate<openvdb::Vec3f, nanovdb::Vec3f>(in_data, voxelSize, out_data, stream);
}

extern "C" void pointToGridFloatToDevice(const HNS::OpenFloatGrid& in_data, const float voxelSize,
                                         nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
                                         const cudaStream_t& stream) {
	pointToGridTemplateToDevice<openvdb::Coord, float, nanovdb::Coord, float, nanovdb::FloatGrid,
	                            nanovdb::SetVoxel<float>>(in_data, voxelSize, handle, stream);
}

extern "C" void pointToGridVectorToDevice(const HNS::OpenVectorGrid& in_data, const float voxelSize,
                                          nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
                                          const cudaStream_t& stream) {
	pointToGridTemplateToDevice<openvdb::Coord, openvdb::Vec3f, nanovdb::Coord, nanovdb::Vec3f, nanovdb::Vec3fGrid,
	                            nanovdb::SetVoxel<nanovdb::Vec3f>>(in_data, voxelSize, handle, stream);
}
