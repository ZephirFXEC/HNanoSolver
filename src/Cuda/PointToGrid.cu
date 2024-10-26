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

	/*
	cudaHostRegister((void*)in_data.pCoords(), npoints * sizeof(nanovdb::Coord), cudaHostRegisterDefault);
	cudaHostRegister((void*)in_data.pValues(), npoints * sizeof(nanovdb::Vec3f), cudaHostRegisterDefault);

	CudaResources<ValueOutT> resources(npoints, stream);

	cudaMemcpyAsync(resources.d_coords, (nanovdb::Coord*)in_data.pCoords(), npoints * sizeof(openvdb::Coord),
					cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(resources.d_values, (ValueOutT*)in_data.pValues(), npoints * sizeof(ValueOutT),
					cudaMemcpyHostToDevice, stream);

	cudaDeviceSynchronize();
	*/

	// Allocate and copy coordinates to the device
	nanovdb::Coord* d_coords = nullptr;
	ValueOutT* d_values = nullptr;
	ValueOutT* d_temp_values = nullptr;

	cudaMalloc(&d_temp_values, npoints * sizeof(ValueOutT));
	cudaMalloc(&d_coords, npoints * sizeof(nanovdb::Coord));
	cudaMemcpy(d_coords, reinterpret_cast<const nanovdb::Coord*>(in_data.pCoords()), npoints * sizeof(nanovdb::Coord),
			   cudaMemcpyHostToDevice);

	cudaMalloc(&d_values, npoints * sizeof(ValueOutT));
	cudaMemcpy(d_values, reinterpret_cast<const ValueOutT*>(in_data.pValues()), npoints * sizeof(ValueOutT),
			   cudaMemcpyHostToDevice);


	auto handle = nanovdb::cudaVoxelsToGrid<ValueOutT>(d_coords, npoints, voxelSize);
	auto* d_grid = handle.template deviceGrid<ValueOutT>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	// Set Grid
	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const auto accessor = d_grid->tree().getAccessor();
		accessor.template set<nanovdb::SetVoxel<ValueOutT>>(d_coords[tid], d_values[tid]);
	});

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		d_temp_values[tid] = d_values[tid];
	});

	//UnloadPointData<ValueOutT>(resources, out_data, npoints, stream);
	// Prepare output and initiate async transfers back to host

	out_data.allocateStandard(npoints);

	// Copy results back to the host
	cudaMemcpy(out_data.pValues(), d_temp_values, sizeof(ValueOutT) * npoints, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_data.pCoords(), d_coords, sizeof(nanovdb::Coord) * npoints, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_coords);
	cudaFree(d_values);
	cudaFree(d_temp_values);

	/*
	out_data.size = npoints;
	out_data.allocateStandard(npoints);

	cudaHostRegister((void*)out_data.pCoords(), npoints * sizeof(nanovdb::Coord), cudaHostRegisterDefault);
	cudaHostRegister((void*)out_data.pValues(), npoints * sizeof(ValueOutT), cudaHostRegisterDefault);

	cudaMemcpyAsync(out_data.pValues(), resources.d_temp_values, npoints * sizeof(ValueOutT), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(out_data.pCoords(), resources.d_coords, npoints * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost, stream);

	cudaHostUnregister((void*)in_data.pCoords());
	cudaHostUnregister((void*)in_data.pValues());
	cudaHostUnregister((void*)out_data.pCoords());
	cudaHostUnregister((void*)out_data.pValues());

	resources.clear(stream);
	*/
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
