#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include "../Utils/GridData.hpp"
#include "PointToGrid.cuh"
#include "Utils.cuh"


__global__ void divergence(const nanovdb::Coord* __restrict__ d_coord, float* __restrict__ d_value, const size_t npoints,
                           const nanovdb::Vec3fGrid* __restrict__ vel) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const auto velAccessor = vel->tree().getAccessor();
	// Linear interpolation sampler
	const auto velSampler = nanovdb::createSampler<1>(velAccessor);
	const float dx = vel->voxelSize()[0];  // voxel spacing

	const nanovdb::Coord coord = d_coord[tid];
	const nanovdb::Vec3f c = coord.asVec3s();

	// Compute neighbor coordinates in short vector form
	const nanovdb::Vec3f cxp1 = c + nanovdb::Vec3f(dx, 0.0f, 0.0f);
	const nanovdb::Vec3f cxm1 = c + nanovdb::Vec3f(-dx, 0.0f, 0.0f);
	const nanovdb::Vec3f cyp1 = c + nanovdb::Vec3f(0.0f, dx, 0.0f);
	const nanovdb::Vec3f cym1 = c + nanovdb::Vec3f(0.0f, -dx, 0.0f);
	const nanovdb::Vec3f czp1 = c + nanovdb::Vec3f(0.0f, 0.0f, dx);
	const nanovdb::Vec3f czm1 = c + nanovdb::Vec3f(0.0f, 0.0f, -dx);

	const nanovdb::Vec3f vel_xm1 = velSampler(cxm1);
	const nanovdb::Vec3f vel_xp1 = velSampler(cxp1);
	const nanovdb::Vec3f vel_ym1 = velSampler(cym1);
	const nanovdb::Vec3f vel_yp1 = velSampler(cyp1);
	const nanovdb::Vec3f vel_zm1 = velSampler(czm1);
	const nanovdb::Vec3f vel_zp1 = velSampler(czp1);

	const float divX = (vel_xp1[0] - vel_xm1[0]) / dx;
	const float divY = (vel_yp1[1] - vel_ym1[1]) / dx;
	const float divZ = (vel_zp1[2] - vel_zm1[2]) / dx;

	d_value[tid] = divX + divY + divZ;
}

__global__ void pressureJacobiIteration(const nanovdb::Coord* __restrict__ d_coords, const size_t npoints,
                                        const nanovdb::FloatGrid* __restrict__ pressureGrid,
                                        const nanovdb::FloatGrid* __restrict__ divergenceGrid,
                                        const nanovdb::FloatGrid* __restrict__ newPressureGrid) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const auto pressureAccessor = pressureGrid->tree().getAccessor();
	const auto divergenceAccessor = divergenceGrid->tree().getAccessor();
	auto newPressureAccessor = newPressureGrid->tree().getAccessor();

	const auto pressureSampler = nanovdb::createSampler<1>(pressureAccessor);
	const auto divergenceSampler = nanovdb::createSampler<1>(divergenceAccessor);

	const nanovdb::Coord coord = d_coords[tid];
	const nanovdb::Vec3f c = coord.asVec3s();
	const float dx = pressureGrid->voxelSize()[0];

	// Neighbor coords
	const nanovdb::Vec3f cxp1 = c + nanovdb::Vec3f(dx, 0.0f, 0.0f);
	const nanovdb::Vec3f cxm1 = c - nanovdb::Vec3f(dx, 0.0f, 0.0f);
	const nanovdb::Vec3f cyp1 = c + nanovdb::Vec3f(0.0f, dx, 0.0f);
	const nanovdb::Vec3f cym1 = c - nanovdb::Vec3f(0.0f, dx, 0.0f);
	const nanovdb::Vec3f czp1 = c + nanovdb::Vec3f(0.0f, 0.0f, dx);
	const nanovdb::Vec3f czm1 = c - nanovdb::Vec3f(0.0f, 0.0f, dx);

	// Neighboring pressures
	const float p_xp1 = pressureSampler(cxp1);
	const float p_xm1 = pressureSampler(cxm1);
	const float p_yp1 = pressureSampler(cyp1);
	const float p_ym1 = pressureSampler(cym1);
	const float p_zp1 = pressureSampler(czp1);
	const float p_zm1 = pressureSampler(czm1);

	// Divergence at coord
	const float div = divergenceSampler(c);

	// Jacobi iteration step
	const float p_new = (p_xp1 + p_xm1 + p_yp1 + p_ym1 + p_zp1 + p_zm1 - div * dx * dx) / 6.0f;

	// Write the new pressure value
	newPressureAccessor.set<nanovdb::SetVoxel<float>>(coord, p_new);
}

__global__ void subtractPressureGradient(const nanovdb::Coord* __restrict__ d_coords, const size_t npoints,
                                         const nanovdb::Vec3fGrid* __restrict__ vel, const nanovdb::FloatGrid* __restrict__ pressureGrid,
                                         const CudaResources<nanovdb::Vec3f, true> out, const float voxelSize) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const auto pressureSampler = nanovdb::createSampler<1>(pressureGrid->getAccessor());
	const auto velSampler = nanovdb::createSampler<1>(vel->getAccessor());

	const nanovdb::Coord coord = d_coords[tid];

	const nanovdb::Vec3f grad = pressureSampler.gradient<>(coord.asVec3s());

	// Update velocity
	nanovdb::Vec3f v = velSampler(coord);
	v -= grad;  //* (1.0f / voxelSize);

	out.d_values[tid] = v;
}


void pressure_projection(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& in_vel, HNS::OpenVectorGrid& in_data,
                         HNS::OpenVectorGrid& out_data, const size_t iteration, const cudaStream_t& stream) {
	using BufferT = nanovdb::CudaDeviceBuffer;

	const size_t npoints = in_data.size;
	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	CudaResources<float, false> div_resources(npoints, stream);
	div_resources.LoadPointCoord(in_data.pCoords(), npoints, stream);

	const nanovdb::Vec3fGrid* in_vel_grid = in_vel.deviceGrid<nanovdb::Vec3f>();
	nanovdb::GridHandle<BufferT> divergence_handle = nanovdb::cudaVoxelsToGrid<float>(div_resources.d_coords, npoints, 0.2f);
	nanovdb::GridHandle<BufferT> pressure_handle_ping = nanovdb::cudaVoxelsToGrid<float>(div_resources.d_coords, npoints, 0.2f);
	nanovdb::GridHandle<BufferT> pressure_handle_pong = nanovdb::cudaVoxelsToGrid<float>(div_resources.d_coords, npoints, 0.2f);


	// Set Divergence Grid
	divergence<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, div_resources.d_values, npoints, in_vel_grid);

	nanovdb::FloatGrid* in_divergence = divergence_handle.deviceGrid<float>();
	set_grid_values<float, nanovdb::FloatTree, false><<<numBlocks, numThreads, 0, stream>>>(div_resources, npoints, in_divergence);

	// Compute Pressure
	nanovdb::FloatGrid* in_pressure_ping = pressure_handle_ping.deviceGrid<float>();
	nanovdb::FloatGrid* in_pressure_pong = pressure_handle_pong.deviceGrid<float>();

	zero_init_grid<<<numBlocks, numThreads, 0, stream>>>(div_resources, npoints, in_pressure_ping);
	zero_init_grid<<<numBlocks, numThreads, 0, stream>>>(div_resources, npoints, in_pressure_pong);

	for (int iter = 0; iter < iteration; ++iter) {
		pressureJacobiIteration<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, npoints, in_pressure_ping, in_divergence,
		                                                              in_pressure_pong);
		std::swap(in_pressure_ping, in_pressure_pong);
	}

	CudaResources<nanovdb::Vec3f, true> vel_resources(npoints, stream);

	// Subtract Pressure Gradient
	subtractPressureGradient<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, npoints, in_vel_grid, in_pressure_ping,
	                                                               vel_resources, 0.2);

	// copy vel back to host
	out_data.allocateCudaPinned(npoints);

	cudaCheck(cudaMemcpy(out_data.pCoords(), div_resources.d_coords, npoints * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(out_data.pValues(), vel_resources.d_values, npoints * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost));

	// Unload data
	vel_resources.cleanup(stream);
	div_resources.cleanup(stream);
	cudaCheckError();
}

extern "C" void PressureProjection(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& in_vel, HNS::OpenVectorGrid& in_data,
                                   HNS::OpenVectorGrid& out_data, const size_t iteration, const cudaStream_t& stream) {
	pressure_projection(in_vel, in_data, out_data, iteration, stream);
}

/*extern "C" void Divergence(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& in_vel, HNS::OpenVectorGrid& in_data,
                           HNS::OpenVectorGrid& out_data, const cudaStream_t& stream) {
    using BufferT = nanovdb::CudaDeviceBuffer;

    const size_t npoints = in_data.size;
    constexpr unsigned int numThreads = 256;
    const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

    CudaResources<float, false> div_resources(npoints, stream);
    div_resources.LoadPointCoord(in_data.pCoords(), npoints, stream);

    const nanovdb::Vec3fGrid* in_vel_grid = in_vel.deviceGrid<nanovdb::Vec3f>();
    nanovdb::GridHandle<BufferT> divergence_handle = nanovdb::cudaVoxelsToGrid<float>(div_resources.d_coords, npoints, 0.2f);

    // Set Divergence Grid
    divergence<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, div_resources.d_values, npoints, in_vel_grid);

    // copy divergence back to host
    out_data.allocateCudaPinned(npoints);
    cudaCheck(cudaMemcpy(out_data.pCoords(), div_resources.d_coords, npoints * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(out_data.pValues(), div_resources.d_values, npoints * sizeof(float), cudaMemcpyDeviceToHost));

    // Unload data
    div_resources.cleanup(stream);
    cudaCheckError();
}*/