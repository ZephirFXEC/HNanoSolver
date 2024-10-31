#include <nanovdb/util/GridHandle.h>
#include "../Utils/GridData.hpp"
#include "Utils.cuh"


extern "C" void ComputeDivergence(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& vel_handle,
                                  HNS::OpenVectorGrid& in_data, HNS::NanoFloatGrid& out_div, const cudaStream_t& stream) {

	const size_t npoints = in_data.size;
	const nanovdb::Vec3fGrid* vel = vel_handle.deviceGrid<nanovdb::Vec3f>();

	cudaCheck(cudaHostRegister(in_data.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));

	nanovdb::Coord* d_coord = nullptr;
	float* d_value = nullptr;
	cudaCheck(cudaMallocAsync(&d_coord, npoints * sizeof(nanovdb::Coord), stream));
	cudaCheck(cudaMallocAsync(&d_value, npoints * sizeof(float), stream));
	cudaCheck(cudaMemcpyAsync(d_coord, in_data.pCoords(), npoints * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream));

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = (npoints + numThreads - 1) / numThreads;

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const auto velAccessor = vel->getAccessor();

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

		d_value[tid] = (divX + divY + divZ) / (2.0f * 0.2f);  // TODO: Hardcoded voxel size
	});
	cudaCheckError();

	out_div.allocateStandard(npoints);

	cudaCheck(cudaHostRegister(out_div.pValues(), npoints * sizeof(float), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(out_div.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));

	cudaCheck(cudaMemcpyAsync(out_div.pValues(), d_value, out_div.size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaCheck(cudaMemcpyAsync(out_div.pCoords(), in_data.pCoords(), out_div.size * sizeof(openvdb::Coord), cudaMemcpyHostToHost, stream));

	cudaCheck(cudaHostUnregister(in_data.pCoords()));
	cudaCheck(cudaHostUnregister(out_div.pValues()));
	cudaCheck(cudaHostUnregister(out_div.pCoords()));

	cudaCheck(cudaFreeAsync(d_coord, stream));
	cudaCheck(cudaFreeAsync(d_value, stream));
}
