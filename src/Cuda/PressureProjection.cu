#include <nanovdb/util/GridHandle.h>

#include "../Utils/GridData.hpp"
#include "PointToGrid.cuh"
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

__global__ void pressureJacobiIteration(const nanovdb::Coord* __restrict__ d_coords, const size_t npoints,
                                        const nanovdb::FloatGrid* __restrict__ pressureGrid,
                                        const nanovdb::FloatGrid* __restrict__ divergenceGrid,
                                        nanovdb::FloatGrid* __restrict__ newPressureGrid) {
	const auto pressureAccessor = pressureGrid->getAccessor();
	const auto divergenceAccessor = divergenceGrid->getAccessor();
	auto newPressureAccessor = newPressureGrid->getAccessor();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const nanovdb::Coord& coord = d_coords[tid];

	// Read neighboring pressure values
	const float p_xp1 = pressureAccessor.getValue(coord.offsetBy(1, 0, 0));
	const float p_xm1 = pressureAccessor.getValue(coord.offsetBy(-1, 0, 0));
	const float p_yp1 = pressureAccessor.getValue(coord.offsetBy(0, 1, 0));
	const float p_ym1 = pressureAccessor.getValue(coord.offsetBy(0, -1, 0));
	const float p_zp1 = pressureAccessor.getValue(coord.offsetBy(0, 0, 1));
	const float p_zm1 = pressureAccessor.getValue(coord.offsetBy(0, 0, -1));

	// Get divergence value at current coord
	const float divergence = divergenceAccessor.getValue(coord);

	// Compute new pressure value
	float p_new = (1.0f / 6.0f) * (p_xp1 + p_xm1 + p_yp1 + p_ym1 + p_zp1 + p_zm1 - divergence);

	// Write new pressure value
	newPressureAccessor.set<nanovdb::SetVoxel<float>>(coord, p_new);
}

__global__ void subtractPressureGradient(const nanovdb::Coord* __restrict__ d_coords, const size_t npoints,
                                         const nanovdb::FloatGrid* __restrict__ pressureGrid, const CudaResources<nanovdb::Vec3f, true> out,
                                         const float voxelSize) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const nanovdb::Coord& coord = d_coords[tid];

	const auto pressureAccessor = pressureGrid->getAccessor();

	// Compute pressure gradient
	const float p_xp1 = pressureAccessor.getValue(coord.offsetBy(1, 0, 0));
	const float p_xm1 = pressureAccessor.getValue(coord.offsetBy(-1, 0, 0));
	const float p_yp1 = pressureAccessor.getValue(coord.offsetBy(0, 1, 0));
	const float p_ym1 = pressureAccessor.getValue(coord.offsetBy(0, -1, 0));
	const float p_zp1 = pressureAccessor.getValue(coord.offsetBy(0, 0, 1));
	const float p_zm1 = pressureAccessor.getValue(coord.offsetBy(0, 0, -1));

	const float gradX = (p_xp1 - p_xm1) / (2.0f * voxelSize);
	const float gradY = (p_yp1 - p_ym1) / (2.0f * voxelSize);
	const float gradZ = (p_zp1 - p_zm1) / (2.0f * voxelSize);

	// Update velocity
	nanovdb::Vec3f vel = out.d_values[tid];

	vel[0] -= gradX;
	vel[1] -= gradY;
	vel[2] -= gradZ;

	out.d_values[tid] = vel;
}


void pressure_projection(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& in_vel, HNS::OpenVectorGrid& in_data,
                         HNS::OpenVectorGrid& out_data, const cudaStream_t& stream) {
	using BufferT = nanovdb::CudaDeviceBuffer;

	const size_t npoints = in_data.size;
	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	cudaCheck(cudaHostRegister(in_data.pCoords(), npoints * sizeof(openvdb::Coord), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(in_data.pValues(), npoints * sizeof(openvdb::Vec3f), cudaHostRegisterDefault));
	cudaCheckError();

	CudaResources<float, false> div_resources(npoints, stream);
	div_resources.LoadPointCoord(in_data.pCoords(), npoints, stream);
	cudaCheckError();

	const nanovdb::Vec3fGrid* in_vel_grid = in_vel.deviceGrid<nanovdb::Vec3f>();

	nanovdb::GridHandle<BufferT> divergence_handle = nanovdb::cudaVoxelsToGrid<float>(div_resources.d_coords, npoints, 0.2);
	cudaCheckError();

	nanovdb::GridHandle<BufferT> pressure_handle_ping = divergence_handle.copy<BufferT>();
	nanovdb::GridHandle<BufferT> pressure_handle_pong = divergence_handle.copy<BufferT>();
	cudaCheckError();

	// Set Divergence Grid
	divergence<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, div_resources.d_values, npoints, in_vel_grid);
	cudaCheckError();
	nanovdb::FloatGrid* in_divergence = divergence_handle.deviceGrid<float>();
	set_grid_values<float, nanovdb::FloatTree, false><<<numBlocks, numThreads, 0, stream>>>(div_resources, npoints, in_divergence);
	cudaCheck(cudaFreeAsync(div_resources.d_values, stream));
	cudaCheckError();


	// Compute Pressure
	nanovdb::FloatGrid* in_pressure_ping = pressure_handle_ping.deviceGrid<float>();
	nanovdb::FloatGrid* in_pressure_pong = pressure_handle_pong.deviceGrid<float>();
	cudaCheckError();

	for (int iter = 0; iter < 20; ++iter) {
		pressureJacobiIteration<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, npoints, in_pressure_ping, in_divergence,
		                                                              in_pressure_pong);
		std::swap(in_pressure_ping, in_pressure_pong);
	}
	cudaCheckError();

	CudaResources<nanovdb::Vec3f, true> vel_resources(npoints, stream);
	vel_resources.LoadPointValue<openvdb::Vec3f>(in_data.pValues(), npoints, stream);
	cudaCheckError();

	// Subtract Pressure Gradient
	subtractPressureGradient<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, npoints, in_pressure_ping, vel_resources, 0.2);
	cudaCheckError();


	// copy vel back to host
	out_data.allocateStandard(npoints);

	cudaCheck(cudaHostRegister(out_data.pCoords(), npoints * sizeof(nanovdb::Coord), cudaHostRegisterDefault));
	cudaCheck(cudaHostRegister(out_data.pValues(), npoints * sizeof(nanovdb::Vec3f), cudaHostRegisterDefault));

	cudaCheck(cudaMemcpy(out_data.pCoords(), div_resources.d_coords, npoints * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(out_data.pValues(), vel_resources.d_values, npoints * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost));

	cudaCheck(cudaHostUnregister(out_data.pCoords()));
	cudaCheck(cudaHostUnregister(out_data.pValues()));

	// Unload data
	vel_resources.cleanup(stream);
	div_resources.cleanup(stream);
	cudaCheckError();
}

extern "C" void PressureProjection(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& in_vel, HNS::OpenVectorGrid& in_data,
                                   HNS::OpenVectorGrid& out_data, const cudaStream_t& stream) {
	pressure_projection(in_vel, in_data, out_data, stream);
}