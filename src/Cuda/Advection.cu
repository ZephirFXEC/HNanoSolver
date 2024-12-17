#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cuda/std/cmath>

#include "../Utils/GridData.hpp"
#include "HNanoGrid/HNanoGrid.cuh"
#include "PointToGrid.cuh"


template <typename T, typename U = std::conditional_t<std::is_same_v<T, float>, nanovdb::FloatTree, nanovdb::Vec3fTree>>
__global__ void advect(const CudaResources<T, true> resources, const size_t npoints, const float dt, const float voxelSize,
                       const nanovdb::Vec3fGrid* __restrict__ vel_grid, const nanovdb::Grid<U>* __restrict__ d_grid) {

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const float inv_voxelSize = 1.0f / voxelSize;
	const float scaled_dt = dt * inv_voxelSize;

	auto accessor = d_grid->tree().getAccessor();
	const auto velAccessor = vel_grid->tree().getAccessor();
	const auto velSampler = nanovdb::createSampler<1>(velAccessor);
	auto valueSampler = nanovdb::createSampler<1>(accessor);

	const nanovdb::Coord ijk = resources.d_coords[tid];
	const T value = resources.d_values[tid];

	if (!accessor.isActive(ijk)) {
		return;
	}

	const nanovdb::Vec3f voxelCoordf = ijk.asVec3s();

	// For MAC velocity sampling, use the helper function defined above
	const nanovdb::Vec3f velocity = sampleMACVelocity(velSampler, voxelCoordf);
	const nanovdb::Vec3f forward_pos = voxelCoordf - velocity * scaled_dt;
	const T value_forward = valueSampler(forward_pos);

	// Backward step for error correction
	const nanovdb::Vec3f back_velocity = sampleMACVelocity(velSampler, forward_pos);
	const nanovdb::Vec3f back_pos = voxelCoordf + back_velocity * scaled_dt;
	const T value_backward = valueSampler(back_pos);

	// Error estimation and correction
	const T error = computeError(value, value_backward);
	T value_corrected = value_forward + error;
	const T max_correction = computeMaxCorrection(value_forward, value);
	value_corrected = clampValue(value_corrected, value_forward - max_correction, value_forward + max_correction);

	constexpr float blend_factor = 0.8f;
	T new_value = lerp(value_forward, value_corrected, blend_factor);

	new_value = enforceNonNegative(new_value);

	// Store the new value
	resources.d_temp_values[tid] = new_value;
}


void advect_points_to_grid_f(HNS::OpenFloatGrid& in_data, const nanovdb::Vec3fGrid* vel_grid, HNS::NanoFloatGrid& out_data,
                             const float voxelSize, const float dt, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<float, true> resources(npoints, stream);

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle;
	pointToTopologyToDevice<float, true>(resources, in_data.pCoords(), npoints, voxelSize, handle, stream);
	fillTopology<float, float, true>(resources, in_data.pValues(), npoints, handle, stream);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	const nanovdb::FloatGrid* d_grid = handle.deviceGrid<float>();
	advect<float><<<numBlocks, numThreads, 0, stream>>>(resources, npoints, dt, voxelSize, vel_grid, d_grid);

	out_data.allocateCudaPinned(npoints);

	resources.UnloadPointData(out_data, stream);

	resources.cleanup(stream);
}

void advect_points_to_grid_v(HNS::OpenVectorGrid& in_data, HNS::NanoVectorGrid& out_data, const float voxelSize, const float dt,
                             const cudaStream_t& stream) {
	const size_t npoints = in_data.size;
	CudaResources<nanovdb::Vec3f, true> resources(npoints, stream);

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle;
	pointToTopologyToDevice<nanovdb::Vec3f, true>(resources, in_data.pCoords(), npoints, voxelSize, handle, stream);
	fillTopology<nanovdb::Vec3f, openvdb::Vec3f, true>(resources, in_data.pValues(), npoints, handle, stream);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	const nanovdb::Vec3fGrid* d_grid = handle.deviceGrid<nanovdb::Vec3f>();
	advect<nanovdb::Vec3f><<<numBlocks, numThreads, 0, stream>>>(resources, npoints, dt, voxelSize, d_grid, d_grid);

	out_data.allocateCudaPinned(npoints);

	resources.UnloadPointData(out_data, stream);

	resources.cleanup(stream);
}


extern "C" void AdvectFloat(HNS::OpenFloatGrid& in_data, const nanovdb::Vec3fGrid* vel_grid, HNS::NanoFloatGrid& out_data,
                            const float voxelSize, const float dt, const cudaStream_t& stream) {
	advect_points_to_grid_f(in_data, vel_grid, out_data, voxelSize, dt, stream);
}

extern "C" void AdvectVector(HNS::OpenVectorGrid& in_data, HNS::NanoVectorGrid& out_data, const float voxelSize, const float dt,
                             const cudaStream_t& stream) {
	advect_points_to_grid_v(in_data, out_data, voxelSize, dt, stream);
}