#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cuda/std/cmath>

#include "../Utils/GridData.hpp"
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include "PointToGrid.cuh"


__global__ void advect(const CudaResources<nanovdb::Coord, float, true> resources, const size_t npoints, const float dt, const float voxelSize,
                       const nanovdb::Vec3fGrid* __restrict__ vel_grid, const nanovdb::FloatGrid* __restrict__ d_grid) {

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const float inv_voxelSize = 1.0f / voxelSize;
	const float scaled_dt = dt * inv_voxelSize;

	const auto accessor = d_grid->tree().getAccessor();
	const auto velAccessor = vel_grid->tree().getAccessor();
	const auto velSampler = nanovdb::math::createSampler<1>(velAccessor);
	const auto valueSampler = nanovdb::math::createSampler<1>(accessor);

	const nanovdb::Coord ijk = resources.d_coords[tid];
	const float original = resources.d_values[tid];

	if (!accessor.isActive(ijk)) {
		return;
	}

	const nanovdb::Vec3f voxelCoordf = ijk.asVec3s();

	// -------------------------------------------
	// Forward step (semi-Lagrangian):
	// Trace backward in time to find donor cell.
	// velocity at voxelCoordf (MAC-sampled)
	const nanovdb::Vec3f velocity = MACToFaceCentered(velSampler, voxelCoordf);
	const nanovdb::Vec3f forward_pos = voxelCoordf - velocity * scaled_dt;
	const float value_forward = valueSampler(forward_pos);


	// -------------------------------------------
	// Backward step for BFECC:
	// From the forward_pos, integrate forward dt again:
	const nanovdb::Vec3f back_velocity = MACToFaceCentered(velSampler, forward_pos);
	const nanovdb::Vec3f back_pos = voxelCoordf + back_velocity * scaled_dt;
	const float value_backward = valueSampler(back_pos);

	// Error estimation and correction
	const float error = computeError(original, value_backward);
	float value_corrected = value_forward + error;
	const float max_correction = computeMaxCorrection(value_forward, original);
	value_corrected = clampValue(value_corrected, value_forward - max_correction, value_forward + max_correction);
	value_corrected = enforceNonNegative(value_corrected);

	// Store the new value
	resources.d_temp_values[tid] = value_corrected;
}

__global__ void advect(const CudaResources<nanovdb::Coord, nanovdb::Vec3f, true> resources, const size_t npoints, const float dt, const float voxelSize,
                       const nanovdb::Vec3fGrid* __restrict__ vel_grid, const nanovdb::Vec3fGrid* __restrict__ d_grid) {

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= npoints) return;

	const float inv_voxelSize = 1.0f / voxelSize;
	const float scaled_dt = dt * inv_voxelSize;

	const auto velAccessor = vel_grid->tree().getAccessor();
	const auto velSampler = nanovdb::math::createSampler<1>(velAccessor);

	const nanovdb::Coord ijk = resources.d_coords[tid];
	const nanovdb::Vec3f original = resources.d_values[tid];

	if (!velAccessor.isActive(ijk)) {
		return;
	}

	const nanovdb::Vec3f voxelCoordf = ijk.asVec3s();

	// -------------------------------------------
	// Forward step (semi-Lagrangian):
	// Trace backward in time to find donor cell.
	// velocity at voxelCoordf (MAC-sampled)
	const nanovdb::Vec3f velocity = MACToFaceCentered(velSampler, voxelCoordf);
	const nanovdb::Vec3f forward_pos = voxelCoordf - velocity * scaled_dt;
	const nanovdb::Vec3f value_forward = MACToFaceCentered(velSampler, forward_pos);


	// -------------------------------------------
	// Backward step for BFECC:
	// From the forward_pos, integrate forward dt again:
	const nanovdb::Vec3f back_velocity = MACToFaceCentered(velSampler, forward_pos);
	const nanovdb::Vec3f back_pos = voxelCoordf + back_velocity * scaled_dt;
	const nanovdb::Vec3f value_backward = MACToFaceCentered(velSampler, back_pos);

	// Error estimation and correction
	const nanovdb::Vec3f error = computeError(original, value_backward);
	nanovdb::Vec3f value_corrected = value_forward + error;
	const nanovdb::Vec3f max_correction = computeMaxCorrection(value_forward, original);
	value_corrected = clampValue(value_corrected, value_forward - max_correction, value_forward + max_correction);

	// Store the new value
	resources.d_temp_values[tid] = value_corrected;
}




void advect_points_to_grids_f(std::vector<HNS::OpenFloatGrid>& in_data, const nanovdb::Vec3fGrid* vel_grid,
                              std::vector<HNS::NanoFloatGrid>& out_data, const float voxelSize, const float dt,
                              const cudaStream_t& stream) {
	const size_t ngrids = in_data.size();

	for (size_t i = 0; i < ngrids; i++) {
		const size_t npoints = in_data[i].size;

		CudaResources<nanovdb::Coord, float, true> resources(npoints, stream);

		nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle;
		pointToTopologyToDevice<float, true>(resources, in_data[i].pCoords(), npoints, voxelSize, handle, stream);
		fillTopology<float, float, true>(resources, in_data[i].pValues(), npoints, handle, stream);

		constexpr unsigned int numThreads = 256;
		const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

		const nanovdb::FloatGrid* d_grid = handle.deviceGrid<float>();
		advect<<<numBlocks, numThreads, 0, stream>>>(resources, npoints, dt, voxelSize, vel_grid, d_grid);

		out_data[i].allocateCudaPinned(npoints);

		resources.UnloadPointData(out_data[i], stream);

		resources.cleanup(stream);
	}
}


void advect_points_to_grid_f(HNS::OpenFloatGrid& in_data, const nanovdb::Vec3fGrid* vel_grid, HNS::NanoFloatGrid& out_data,
                             const float voxelSize, const float dt, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<nanovdb::Coord, float, true> resources(npoints, stream);

	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle;
	pointToTopologyToDevice<float, true>(resources, in_data.pCoords(), npoints, voxelSize, handle, stream);
	fillTopology<float, float, true>(resources, in_data.pValues(), npoints, handle, stream);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	const nanovdb::FloatGrid* d_grid = handle.deviceGrid<float>();
	advect<<<numBlocks, numThreads, 0, stream>>>(resources, npoints, dt, voxelSize, vel_grid, d_grid);

	out_data.allocateCudaPinned(npoints);

	resources.UnloadPointData(out_data, stream);

	resources.cleanup(stream);
}

void advect_points_to_grid_v(HNS::OpenVectorGrid& in_data, HNS::NanoVectorGrid& out_data, const float voxelSize, const float dt,
                             const cudaStream_t& stream) {
	const size_t npoints = in_data.size;
	CudaResources<nanovdb::Coord, nanovdb::Vec3f, true> resources(npoints, stream);

	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle;
	pointToTopologyToDevice<nanovdb::Vec3f, true>(resources, in_data.pCoords(), npoints, voxelSize, handle, stream);
	fillTopology<nanovdb::Vec3f, openvdb::Vec3f, true>(resources, in_data.pValues(), npoints, handle, stream);

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	const nanovdb::Vec3fGrid* d_grid = handle.deviceGrid<nanovdb::Vec3f>();
	advect<<<numBlocks, numThreads, 0, stream>>>(resources, npoints, dt, voxelSize, d_grid, d_grid);

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

extern "C" void AdvectFloats(std::vector<HNS::OpenFloatGrid>& in_data, const nanovdb::Vec3fGrid* vel_grid,
                             std::vector<HNS::NanoFloatGrid>& out_data, const float voxelSize, const float dt, const cudaStream_t& stream) {
	advect_points_to_grids_f(in_data, vel_grid, out_data, voxelSize, dt, stream);
}