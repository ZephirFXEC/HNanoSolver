#include <cuda/std/__algorithm/clamp.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cuda/std/cmath>

#include "../Utils/GridData.hpp"
#include "HNanoGrid/HNanoGrid.cuh"
#include "Utils.cuh"


extern "C" void advect_points_to_grid_f(const HNS::OpenFloatGrid& in_data, const nanovdb::Vec3fGrid* vel_grid,
                                        HNS::NanoFloatGrid& out_data, const float voxelSize, const float dt,
                                        const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<float> resources(npoints, stream);
	HostMemoryManager<float, float> memory_manager(in_data, out_data);
	LoadPointData<float, float>(resources, in_data, npoints, stream);

	// Generate a NanoVDB grid that contains the list of voxels on the device
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle =
	    nanovdb::cudaVoxelsToGrid<float>(resources.d_coords, npoints, voxelSize);
	nanovdb::FloatGrid* d_grid = handle.deviceGrid<float>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const auto accessor = d_grid->tree().getAccessor();
		accessor.set<nanovdb::SetVoxel<float>>(resources.d_coords[tid], resources.d_values[tid]);
	});

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const nanovdb::Coord& ijk = resources.d_coords[tid];
		const float& density = resources.d_values[tid];

		const auto accessor = d_grid->tree().getAccessor();

		if (accessor.isActive(ijk)) {
			const auto velAccessor = vel_grid->tree().getAccessor();
			const auto velSampler = nanovdb::createSampler<1>(velAccessor);
			const auto denSampler = nanovdb::createSampler<1>(accessor);

			const nanovdb::Vec3f voxelCoordf = ijk.asVec3s();
			const float inv_voxelSize = 1.0f / voxelSize;

			// Forward step
			const nanovdb::Vec3f velocity = velSampler(voxelCoordf);
			const nanovdb::Vec3f forward_pos = voxelCoordf - velocity * (dt * inv_voxelSize);
			const float d_forward = denSampler(forward_pos);

			// Backward step
			const nanovdb::Vec3f back_pos = voxelCoordf + velSampler(forward_pos) * (dt * inv_voxelSize);
			const float d_backward = denSampler(back_pos);

			// Error estimation and correction
			const float error = 0.5f * (density - d_backward);
			float d_corrected = d_forward + error;

			// Limit the correction based on the neighborhood of the forward position
			const float max_correction = 0.5f * fabsf(d_forward - density);
			d_corrected = __saturatef((d_corrected - d_forward + max_correction) * (1.0f / (2.0f * max_correction))) *
			                  (2.0f * max_correction) +
			              d_forward - max_correction;

			// Final advection (blend between semi-Lagrangian and BFECC result)
			constexpr float blend_factor = 0.8f;
			float new_density = __fmaf_rn(blend_factor, d_corrected - d_forward, d_forward);

			// Ensure non-negativity
			new_density = fmaxf(0.0f, new_density);

			resources.d_temp_values[tid] = new_density;
		}
	});

	UnloadPointData<float>(resources, out_data, npoints, stream);

	resources.clear(stream);
}


extern "C" void advect_points_to_grid_v(const HNS::OpenVectorGrid& in_data, HNS::NanoVectorGrid& out_data,
                                        const float voxelSize, const float dt, const cudaStream_t& stream) {
	const size_t npoints = in_data.size;

	CudaResources<nanovdb::Vec3f> resources(npoints, stream);
	HostMemoryManager<openvdb::Vec3f, nanovdb::Vec3f> memory_manager(in_data, out_data);
	LoadPointData<openvdb::Vec3f, nanovdb::Vec3f>(resources, in_data, npoints, stream);

	// Generate a NanoVDB grid that contains the list of voxels on the device
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle =
	    nanovdb::cudaVoxelsToGrid<nanovdb::Vec3f>(resources.d_coords, npoints, voxelSize);
	nanovdb::Vec3fGrid* d_grid = handle.deviceGrid<nanovdb::Vec3f>();

	constexpr unsigned int numThreads = 256;
	const unsigned int numBlocks = blocksPerGrid(npoints, numThreads);

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const auto accessor = d_grid->tree().getAccessor();
		accessor.set<nanovdb::SetVoxel<nanovdb::Vec3f>>(resources.d_coords[tid], resources.d_values[tid]);
	});
	cudaCheckError();

	lambdaKernel<<<numBlocks, numThreads, 0, stream>>>(npoints, [=] __device__(const size_t tid) {
		const nanovdb::Coord& ijk = resources.d_coords[tid];
		const nanovdb::Vec3f& velocity = resources.d_values[tid];
		const auto velAccessor = d_grid->tree().getAccessor();

		const auto velSampler = nanovdb::createSampler<1>(velAccessor);

		const float inv_voxelSize = 1.0f / voxelSize;
		const nanovdb::Vec3f voxelCoordf = ijk.asVec3s();
		const nanovdb::Vec3f scaled_dt_velocity = velocity * (dt * inv_voxelSize);

		// Perform forward and backward advection using velocity
		const nanovdb::Vec3f forward_pos = voxelCoordf - scaled_dt_velocity;
		const nanovdb::Vec3f backward_pos = voxelCoordf + scaled_dt_velocity;

		const nanovdb::Vec3f v_forward = velSampler(forward_pos);
		const nanovdb::Vec3f v_backward = velSampler(backward_pos);

		// Error estimation and correction
		const nanovdb::Vec3f error = 0.5f * (velocity - v_backward);
		nanovdb::Vec3f v_corrected = v_forward + error;

		nanovdb::Vec3f max_correction;
		max_correction[0] = cuda::std::abs(0.5f * (v_forward[0] - velocity[0]));
		max_correction[1] = cuda::std::abs(0.5f * (v_forward[1] - velocity[1]));
		max_correction[2] = cuda::std::abs(0.5f * (v_forward[2] - velocity[2]));

		v_corrected[0] =
		    cuda::std::clamp(v_corrected[0], v_forward[0] - max_correction[0], v_forward[0] + max_correction[0]);
		v_corrected[1] =
		    cuda::std::clamp(v_corrected[1], v_forward[1] - max_correction[1], v_forward[1] + max_correction[1]);
		v_corrected[2] =
		    cuda::std::clamp(v_corrected[2], v_forward[2] - max_correction[2], v_forward[2] + max_correction[2]);


		constexpr float blend_factor = 0.8f;  // Adjust this value between 0 and 1
		nanovdb::Vec3f new_velocity;
		new_velocity[0] = lerp(v_forward[0], v_corrected[0], blend_factor);
		new_velocity[1] = lerp(v_forward[1], v_corrected[1], blend_factor);
		new_velocity[2] = lerp(v_forward[2], v_corrected[2], blend_factor);

		// Store the new velocity
		resources.d_temp_values[tid] = new_velocity;
	});
	cudaCheckError();

	UnloadPointData<nanovdb::Vec3f>(resources, out_data, npoints, stream);

	resources.clear(stream);
}