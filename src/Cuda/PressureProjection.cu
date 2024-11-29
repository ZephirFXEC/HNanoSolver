#include <nanovdb/util/GridHandle.h>

#include "../Utils/GridData.hpp"
#include "Utils.cuh"


__global__ void divergence(const nanovdb::Coord* __restrict__ d_coord, float* __restrict__ d_value, const size_t npoints,
                           const nanovdb::Vec3fGrid* __restrict__ vel) {
	const auto velAccessor = vel->getAccessor();

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const nanovdb::Coord& coord = d_coord[tid];

	// Fetch neighbors to compute central differences
	nanovdb::Vec3f vel_xm1 = velAccessor.getValue(coord - nanovdb::Coord(1, 0, 0));
	nanovdb::Vec3f vel_xp1 = velAccessor.getValue(coord + nanovdb::Coord(1, 0, 0));
	nanovdb::Vec3f vel_ym1 = velAccessor.getValue(coord - nanovdb::Coord(0, 1, 0));
	nanovdb::Vec3f vel_yp1 = velAccessor.getValue(coord + nanovdb::Coord(0, 1, 0));
	nanovdb::Vec3f vel_zm1 = velAccessor.getValue(coord - nanovdb::Coord(0, 0, 1));
	nanovdb::Vec3f vel_zp1 = velAccessor.getValue(coord + nanovdb::Coord(0, 0, 1));

	// Compute central differences for divergence
	const float divX = vel_xp1[0] - vel_xm1[0];
	const float divY = vel_yp1[1] - vel_ym1[1];
	const float divZ = vel_zp1[2] - vel_zm1[2];

	d_value[tid] = (divX + divY + divZ) / (2.0f * vel->voxelSize()[0]);
}

extern "C" void ComputeDivergence(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& vel_handle, HNS::OpenVectorGrid& in_data,
                                  HNS::NanoFloatGrid& out_div, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;
	const nanovdb::Vec3fGrid* vel = vel_handle.deviceGrid<nanovdb::Vec3f>();

	cudaCheck(cudaHostRegister(in_data.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));

	const CudaResources<float, false> resources(npoints, stream);

	resources.LoadPointCoord(in_data.pCoords(), npoints, stream);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = (npoints + numThreads - 1) / numThreads;

	divergence<<<numBlocks, numThreads, 0, stream>>>(resources.d_coords, resources.d_values, npoints, vel);

	cudaCheckError();

	out_div.allocateStandard(npoints);

	cudaCheck(cudaHostRegister(out_div.pValues(), npoints * sizeof(float), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(out_div.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));

	cudaCheck(cudaMemcpyAsync(out_div.pValues(), resources.d_values, out_div.size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaCheck(cudaMemcpyAsync(out_div.pCoords(), in_data.pCoords(), out_div.size * sizeof(openvdb::Coord), cudaMemcpyHostToHost, stream));

	cudaCheck(cudaHostUnregister(in_data.pCoords()));
	cudaCheck(cudaHostUnregister(out_div.pValues()));
	cudaCheck(cudaHostUnregister(out_div.pCoords()));

	cudaCheck(cudaFreeAsync(resources.d_coords, stream));
	cudaCheck(cudaFreeAsync(resources.d_values, stream));
}


extern "C" void ComputePressure(HNS::NanoFloatGrid& in_divergence, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& vel_handle,
                                const cudaStream_t& stream) {
	/*
	const size_t npoints = in_divergence.size;
	const float* d_divergence = in_divergence.pValues();
	const nanovdb::Coord* d_coords = in_divergence.pCoords();

	nanovdb::FloatGrid* pressure = vel_handle.copy()[0];

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = (npoints + numThreads - 1) / numThreads;

	for (int i = 0; i < 100; ++i) {  // Iterate for a fixed number of iterations or until convergence
	    lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
	        const auto accessor = pressure_grid->tree().getAccessor();
	        const nanovdb::Coord& coord = d_coords[tid];

	        const float p_xm1 = accessor.getValue(coord - nanovdb::Coord(1, 0, 0));
	        const float p_xp1 = accessor.getValue(coord + nanovdb::Coord(1, 0, 0));
	        const float p_ym1 = accessor.getValue(coord - nanovdb::Coord(0, 1, 0));
	        const float p_yp1 = accessor.getValue(coord + nanovdb::Coord(0, 1, 0));
	        const float p_zm1 = accessor.getValue(coord - nanovdb::Coord(0, 0, 1));
	        const float p_zp1 = accessor.getValue(coord + nanovdb::Coord(0, 0, 1));

	        const float divergence = d_divergence[tid];

	        temp_pressure[tid] = (p_xm1 + p_xp1 + p_ym1 + p_yp1 + p_zm1 + p_zp1 - divergence) / 6.0f;
	    });
	    std::swap(pressure_grid, d_new_pressure);
	}
	*/
}