#include <nanovdb/util/GridHandle.h>

#include "../Utils/GridData.hpp"
#include "Utils.cuh"


extern "C" void ComputeDivergence(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& vel_handle,
                                  HNS::NanoFloatGrid& out_div, const cudaStream_t& stream) {
	const nanovdb::Vec3fGrid* vel = vel_handle.deviceGrid<nanovdb::Vec3f>();


	nanovdb::Coord* d_coord = nullptr;
	float* d_value = nullptr;
	cudaMalloc(&d_coord, out_div.size * sizeof(nanovdb::Coord));
	cudaMalloc(&d_value, out_div.size * sizeof(float));
	cudaMemcpy(d_coord, out_div.pCoords, out_div.size * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = (out_div.size + numThreads - 1) / numThreads;

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(out_div.size, [=] __device__(const size_t tid) {
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

	cudaMemcpy(out_div.pValues, d_value, out_div.size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_coord);
	cudaFree(d_value);
}
