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
	const nanovdb::Vec3f c = coord.asVec3s(); // This is in index space

	// For MAC, sample u at (i+0.5,j,k) and (i-0.5,j,k)
	const nanovdb::Vec3f u_ip = c + nanovdb::Vec3f(0.5f, 0.0f, 0.0f); // u at i+1/2
	const nanovdb::Vec3f u_im = c + nanovdb::Vec3f(-0.5f, 0.0f, 0.0f);// u at i-1/2

	// v at (i,j+0.5,k) and (i,j-0.5,k)
	const nanovdb::Vec3f v_jp = c + nanovdb::Vec3f(0.0f, 0.5f, 0.0f); // v at j+1/2
	const nanovdb::Vec3f v_jm = c + nanovdb::Vec3f(0.0f,-0.5f, 0.0f); // v at j-1/2

	// w at (i,j,k+0.5) and (i,j,k-0.5)
	const nanovdb::Vec3f w_kp = c + nanovdb::Vec3f(0.0f, 0.0f, 0.5f); // w at k+1/2
	const nanovdb::Vec3f w_km = c + nanovdb::Vec3f(0.0f, 0.0f,-0.5f); // w at k-1/2

	// Sample velocities
	// u component is stored in vel.x, v in vel.y, w in vel.z
	const float u_i_p = velSampler(u_ip)[0];
	const float u_i_m = velSampler(u_im)[0];

	const float v_j_p = velSampler(v_jp)[1];
	const float v_j_m = velSampler(v_jm)[1];

	const float w_k_p = velSampler(w_kp)[2];
	const float w_k_m = velSampler(w_km)[2];

	// Divergence: (u(i+1/2)-u(i-1/2) + v(j+1/2)-v(j-1/2) + w(k+1/2)-w(k-1/2)) / dx
	const float divX = (u_i_p - u_i_m) / dx;
	const float divY = (v_j_p - v_j_m) / dx;
	const float divZ = (w_k_p - w_k_m) / dx;

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
	const nanovdb::Vec3f c = coord.asVec3s();  // index-space coordinates
	const float dx = pressureGrid->voxelSize()[0];

	// Neighbor coords in index space
	const nanovdb::Vec3f cxp1 = c + nanovdb::Vec3f( 1.0f, 0.0f, 0.0f);
	const nanovdb::Vec3f cxm1 = c + nanovdb::Vec3f(-1.0f, 0.0f, 0.0f);
	const nanovdb::Vec3f cyp1 = c + nanovdb::Vec3f(0.0f,  1.0f, 0.0f);
	const nanovdb::Vec3f cym1 = c + nanovdb::Vec3f(0.0f, -1.0f, 0.0f);
	const nanovdb::Vec3f czp1 = c + nanovdb::Vec3f(0.0f, 0.0f,  1.0f);
	const nanovdb::Vec3f czm1 = c + nanovdb::Vec3f(0.0f, 0.0f, -1.0f);

	// Neighboring pressures using the sampler
	const float p_xp1 = pressureSampler(cxp1);
	const float p_xm1 = pressureSampler(cxm1);
	const float p_yp1 = pressureSampler(cyp1);
	const float p_ym1 = pressureSampler(cym1);
	const float p_zp1 = pressureSampler(czp1);
	const float p_zm1 = pressureSampler(czm1);

	// Divergence at c
	const float div = divergenceSampler(c);

	// Jacobi iteration step: (sum_of_neighbors - div * dx^2) / 6
	const float p_new = (p_xp1 + p_xm1 + p_yp1 + p_ym1 + p_zp1 + p_zm1 - div * dx * dx) / 6.0f;

	// Write the new pressure value
	newPressureAccessor.set<nanovdb::SetVoxel<float>>(coord, p_new);
}

__global__ void subtractPressureGradient(const nanovdb::Coord* __restrict__ d_coords, const size_t npoints,
                                         const nanovdb::Vec3fGrid* __restrict__ vel, const nanovdb::FloatGrid* __restrict__ pressureGrid,
                                         const CudaResources<nanovdb::Vec3f, true> out, const float voxelSize) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const auto pressureAccessor = pressureGrid->tree().getAccessor();
	const auto velAccessor = vel->tree().getAccessor();

	const auto pressureSampler = nanovdb::createSampler<1>(pressureAccessor);
	const auto velSampler = nanovdb::createSampler<1>(velAccessor);

	// Assume a time step dt = 1/24
	const float dt = 1.0f / 24.0f;
	const float dx = voxelSize;

	const nanovdb::Vec3f coord = d_coords[tid].asVec3s();

	// Sample pressures at neighboring cells
	const float p_xp1 = pressureSampler(coord + nanovdb::Vec3f( 1.0f, 0.0f, 0.0f));
	const float p_xm1 = pressureSampler(coord + nanovdb::Vec3f(-1.0f, 0.0f, 0.0f));
	const float p_yp1 = pressureSampler(coord + nanovdb::Vec3f(0.0f,  1.0f, 0.0f));
	const float p_ym1 = pressureSampler(coord + nanovdb::Vec3f(0.0f, -1.0f, 0.0f));
	const float p_zp1 = pressureSampler(coord + nanovdb::Vec3f(0.0f, 0.0f,  1.0f));
	const float p_zm1 = pressureSampler(coord + nanovdb::Vec3f(0.0f, 0.0f, -1.0f));

	// Compute the pressure gradient using central differences in index space,
	// then convert to world space by dividing by dx.
	const float gradX = (p_xp1 - p_xm1) / (2.0f * dx);
	const float gradY = (p_yp1 - p_ym1) / (2.0f * dx);
	const float gradZ = (p_zp1 - p_zm1) / (2.0f * dx);

	nanovdb::Vec3f v = velSampler(coord);

	// Update velocity using the time step dt:
	// v_new = v - dt * ∇p (assuming density = 1 for simplicity)
	v[0] -= gradX;
	v[1] -= gradY;
	v[2] -= gradZ;

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