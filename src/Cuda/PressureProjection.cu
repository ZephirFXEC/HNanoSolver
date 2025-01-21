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
	const nanovdb::Vec3f c = coord.asVec3s();  // This is in index space

	// to get the face-centered velocities along each axis.
	const float u_i_p = sampleMACVelocity(velSampler, c + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float u_i_m = sampleMACVelocity(velSampler, c + nanovdb::Vec3f(-0.5f, 0.0f, 0.0f))[0];

	const float v_j_p = sampleMACVelocity(velSampler, c + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float v_j_m = sampleMACVelocity(velSampler, c + nanovdb::Vec3f(0.0f, -0.5f, 0.0f))[1];

	const float w_k_p = sampleMACVelocity(velSampler, c + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	const float w_k_m = sampleMACVelocity(velSampler, c + nanovdb::Vec3f(0.0f, 0.0f, -0.5f))[2];

	// Divergence: (u(i+1/2)-u(i-1/2) + v(j+1/2)-v(j-1/2) + w(k+1/2)-w(k-1/2)) / dx
	const float divX = (u_i_p - u_i_m) / dx;
	const float divY = (v_j_p - v_j_m) / dx;
	const float divZ = (w_k_p - w_k_m) / dx;

	d_value[tid] = divX + divY + divZ;
}

__global__ void pressureJacobiIteration(const nanovdb::Coord* __restrict__ d_coords, size_t npoints,
                                        const nanovdb::FloatGrid* __restrict__ pressureGrid,
                                        const nanovdb::FloatGrid* __restrict__ divergenceGrid,
                                        const nanovdb::FloatGrid* __restrict__ newPressureGrid) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	// Accessors
	const auto pressureAccessor = pressureGrid->tree().getAccessor();
	const auto divergenceAccessor = divergenceGrid->tree().getAccessor();
	const auto newPressureAccessor = newPressureGrid->tree().getAccessor();

	// Samplers
	const auto pressureSampler = nanovdb::createSampler<1>(pressureAccessor);
	const auto divergenceSampler = nanovdb::createSampler<1>(divergenceAccessor);

	// Current cell's coordinate
	const nanovdb::Coord coord = d_coords[tid];
	const nanovdb::Vec3f c = coord.asVec3s();  // index space
	const float dx = pressureGrid->voxelSize()[0];

	// Offsets in index space
	const nanovdb::Vec3f cxp1 = c + nanovdb::Vec3f(1.0f, 0.0f, 0.0f);
	const nanovdb::Vec3f cxm1 = c + nanovdb::Vec3f(-1.0f, 0.0f, 0.0f);
	const nanovdb::Vec3f cyp1 = c + nanovdb::Vec3f(0.0f, 1.0f, 0.0f);
	const nanovdb::Vec3f cym1 = c + nanovdb::Vec3f(0.0f, -1.0f, 0.0f);
	const nanovdb::Vec3f czp1 = c + nanovdb::Vec3f(0.0f, 0.0f, 1.0f);
	const nanovdb::Vec3f czm1 = c + nanovdb::Vec3f(0.0f, 0.0f, -1.0f);

	// Neighbor pressures
	const float p_xp1 = pressureSampler(cxp1);
	const float p_xm1 = pressureSampler(cxm1);
	const float p_yp1 = pressureSampler(cyp1);
	const float p_ym1 = pressureSampler(cym1);
	const float p_zp1 = pressureSampler(czp1);
	const float p_zm1 = pressureSampler(czm1);

	// Divergence at center
	const float div = divergenceSampler(c);

	// Jacobi iteration step (standard 6-neighbor Laplacian):
	// p_new = ( sum_of_neighbors - div * dx^2 ) / 6
	const float p_new = (p_xp1 + p_xm1 + p_yp1 + p_ym1 + p_zp1 + p_zm1 - div * dx * dx) / 6.0f;

	// Write the new pressure
	newPressureAccessor.set<nanovdb::SetVoxel<float>>(coord, p_new);
}


__global__ void subtractPressureGradient(const nanovdb::Coord* __restrict__ d_coords, size_t npoints,
                                         const nanovdb::Vec3fGrid* __restrict__ velGrid,       // velocity at faces
                                         const nanovdb::FloatGrid* __restrict__ pressureGrid,  // pressure at cell centers
                                         CudaResources<nanovdb::Vec3f, true> out, float voxelSize) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints) return;

    // Accessors / Samplers
    const auto pressureAccessor = pressureGrid->tree().getAccessor();
    const auto velAccessor = velGrid->tree().getAccessor();
    const auto pressureSampler = nanovdb::createSampler<1>(pressureAccessor);
    const auto velSampler = nanovdb::createSampler<1>(velAccessor);

    const float dx = voxelSize;

    // The cell center coordinate
    const nanovdb::Vec3f c = d_coords[tid].asVec3s();
    nanovdb::Vec3f v = sampleMACVelocity(velSampler, c);

    // For u component: Sample velocity at (i+1/2,j,k) relative to cell center
    {
        // For x-component, we're already at the face center
        const float p_left = pressureSampler(c);                                    // p(i,j,k)
        const float p_right = pressureSampler(c + nanovdb::Vec3f(1, 0.0f, 0.0f)); // p(i+1,j,k)
        v[0] -= (p_right - p_left) / dx;
    }

    // For v component: Sample velocity at (i,j+1/2,k) relative to cell center
    {
        const float p_bottom = pressureSampler(c);                                    // p(i,j,k)
        const float p_top = pressureSampler(c + nanovdb::Vec3f(0.0f, 1, 0.0f));     // p(i,j+1,k)
        v[1] -= (p_top - p_bottom) / dx;
    }

    // For w component: Sample velocity at (i,j,k+1/2) relative to cell center
    {
        const float p_back = pressureSampler(c);                                    // p(i,j,k)
        const float p_front = pressureSampler(c + nanovdb::Vec3f(0.0f, 0.0f, 1)); // p(i,j,k+1)
        v[2] -= (p_front - p_back) / dx;
    }

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

extern "C" void Divergence(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& in_vel, HNS::OpenVectorGrid& in_data,
                           HNS::OpenFloatGrid& out_data, const cudaStream_t& stream) {
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
}