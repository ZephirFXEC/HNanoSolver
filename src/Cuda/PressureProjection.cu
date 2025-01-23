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
	const float dx = vel->voxelSize()[0];

	const nanovdb::Coord coord = d_coord[tid];
	const nanovdb::Vec3f c = coord.asVec3s();

	const float xp = velSampler(c + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float xm = velSampler(c - nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float yp = velSampler(c + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float ym = velSampler(c - nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float zp = velSampler(c + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	const float zm = velSampler(c - nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	const float dixX = (xp - xm) / dx;
	const float dixY = (yp - ym) / dx;
	const float dixZ = (zp - zm) / dx;

	d_value[tid] = dixX + dixY + dixZ;
}

__global__ void redBlackGaussSeidelUpdate(
	const nanovdb::Coord* __restrict__ d_coords,
	const size_t npoints,
	nanovdb::FloatGrid* __restrict__ pressureGrid,
	const nanovdb::FloatGrid* __restrict__ divergenceGrid,
	const float dx,
	const int color,
	const float omega)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	nanovdb::Coord c = d_coords[tid];
	const int i = c.x(), j = c.y(), k = c.z();

	// skip if not the correct color
	if (((i + j + k) & 1) != color) return;

	const auto pAcc  = pressureGrid->tree().getAccessor();
	const auto divAcc= divergenceGrid->tree().getAccessor();

	// gather neighbors (assuming in range)
	const float pxp1 = pAcc.getValue(nanovdb::Coord(i+1,j,k));
	const float pxm1 = pAcc.getValue(nanovdb::Coord(i-1,j,k));
	const float pyp1 = pAcc.getValue(nanovdb::Coord(i,j+1,k));
	const float pym1 = pAcc.getValue(nanovdb::Coord(i,j-1,k));
	const float pzp1 = pAcc.getValue(nanovdb::Coord(i,j,k+1));
	const float pzm1 = pAcc.getValue(nanovdb::Coord(i,j,k-1));

	const float divVal = divAcc.getValue(c);

	// Standard 6-neighbor Laplacian-based update
	const float pOld = pAcc.getValue(c);
	const float pGS  = (pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1 - divVal*dx*dx) / 6.0f;

	// SOR step
	float pNew = pOld + omega*(pGS - pOld);

	// in-place update
	pAcc.set<nanovdb::SetVoxel<float>>(c, pNew);
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
	const nanovdb::Vec3f vel = sampleMACVelocity(velSampler, c);
	nanovdb::Vec3f v;

	// For u component: Sample velocity at (i+1/2,j,k) relative to cell center
	{
		// For x-component, we're already at the face center
		const float p_left = pressureSampler(c);                                   // p(i,j,k)
		const float p_right = pressureSampler(c + nanovdb::Vec3f(1, 0.0f, 0.0f));  // p(i+1,j,k)
		v[0] = vel[0] - (p_right - p_left) / dx;
	}

	// For v component: Sample velocity at (i,j+1/2,k) relative to cell center
	{
		const float p_bottom = pressureSampler(c);                               // p(i,j,k)
		const float p_top = pressureSampler(c + nanovdb::Vec3f(0.0f, 1, 0.0f));  // p(i,j+1,k)
		v[1] = vel[1] - (p_top - p_bottom) / dx;
	}

	// For w component: Sample velocity at (i,j,k+1/2) relative to cell center
	{
		const float p_back = pressureSampler(c);                                   // p(i,j,k)
		const float p_front = pressureSampler(c + nanovdb::Vec3f(0.0f, 0.0f, 1));  // p(i,j,k+1)
		v[2] = vel[2] - (p_front - p_back) / dx;
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
	nanovdb::GridHandle<BufferT> pressure_handle = nanovdb::cudaVoxelsToGrid<float>(div_resources.d_coords, npoints, 0.2f);


	// Set Divergence Grid
	divergence<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, div_resources.d_values, npoints, in_vel_grid);

	nanovdb::FloatGrid* in_divergence = divergence_handle.deviceGrid<float>();
	set_grid_values<float, nanovdb::FloatTree, false><<<numBlocks, numThreads, 0, stream>>>(div_resources, npoints, in_divergence);

	// Compute Pressure
	nanovdb::FloatGrid* in_pressure = pressure_handle.deviceGrid<float>();

	zero_init_grid<<<numBlocks, numThreads, 0, stream>>>(div_resources, npoints, in_pressure);

	for (int iter = 0; iter < iteration; iter++) {
		// Red update
		redBlackGaussSeidelUpdate<<<numBlocks, numThreads, 0, stream>>>(
			div_resources.d_coords, npoints, in_pressure, in_divergence, 0.2, /*color=*/0, 1.9);

		cudaDeviceSynchronize();

		// Black update
		redBlackGaussSeidelUpdate<<<numBlocks, numThreads, 0, stream>>>(
			div_resources.d_coords, npoints, in_pressure, in_divergence, 0.2, /*color=*/1, 1.9);

		cudaDeviceSynchronize();
	}

	CudaResources<nanovdb::Vec3f, true> vel_resources(npoints, stream);

	// Subtract Pressure Gradient
	subtractPressureGradient<<<numBlocks, numThreads, 0, stream>>>(div_resources.d_coords, npoints, in_vel_grid, in_pressure,
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