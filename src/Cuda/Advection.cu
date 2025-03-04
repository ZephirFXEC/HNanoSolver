#include <openvdb/Types.h>

#include <cuda/std/cmath>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Utils.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

__global__ void advect_idx(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ coords,
                           const nanovdb::Vec3f* __restrict__ velocityData, const float* __restrict__ inData, float* __restrict__ outData,
                           const size_t totalVoxels, const float dt, const float voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float inv_voxelSize = 1.0f / voxelSize;
	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto dataSampler = IndexSampler<float, 1>(idxSampler, inData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f pos = coord.asVec3s();

	const float original = dataSampler(coord);
	// -------------------------------------------
	// Forward step (semi-Lagrangian):
	// Trace backward in time to find donor cell.
	// velocity at voxelCoordf (MAC-sampled)
	const nanovdb::Vec3f velocity = velocitySampler(pos);
	const nanovdb::Vec3f forward_pos = pos - velocity * scaled_dt;
	const float value_forward = dataSampler(forward_pos);


	// -------------------------------------------
	// Backward step for BFECC:
	// From the forward_pos, integrate forward dt again:
	const nanovdb::Vec3f back_velocity = velocitySampler(forward_pos);
	const nanovdb::Vec3f back_pos = pos + back_velocity * scaled_dt;
	const float value_backward = dataSampler(back_pos);


	// Error estimation and correction
	const float error = computeError(original, value_backward);
	float value_corrected = value_forward + error;
	const float max_correction = computeMaxCorrection(value_forward, original);
	value_corrected = clampValue(value_corrected, value_forward - max_correction, value_forward + max_correction);
	value_corrected = enforceNonNegative(value_corrected);


	// Store the new value
	outData[idx] = value_corrected;
}


__global__ void advect_idx(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ coords,
                           const nanovdb::Vec3f* __restrict__ velocityData, nanovdb::Vec3f* __restrict__ outVelocity,
                           const size_t totalVoxels, const float dt, const float voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float inv_voxelSize = 1.0f / voxelSize;
	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f pos = coord.asVec3s();

	const nanovdb::Vec3f velocity = velocitySampler(coord);
	const nanovdb::Vec3f forward_pos = pos - velocity * scaled_dt;
	const nanovdb::Vec3f value_forward = velocitySampler(forward_pos);

	// -------------------------------------------
	// Backward step for BFECC:
	// From the forward_pos, integrate forward dt again:
	const nanovdb::Vec3f back_velocity = velocitySampler(forward_pos);
	const nanovdb::Vec3f back_pos = pos + back_velocity * scaled_dt;
	const nanovdb::Vec3f value_backward = velocitySampler(back_pos);

	// Error estimation and correction
	const nanovdb::Vec3f error = computeError(velocity, value_backward);
	nanovdb::Vec3f value_corrected = value_forward + error;
	const nanovdb::Vec3f max_correction = computeMaxCorrection(value_forward, velocity);
	value_corrected = clampValue(value_corrected, value_forward - max_correction, value_forward + max_correction);

	// Store the new value
	outVelocity[idx] = value_corrected;
}


void advect_index_grid(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& mainStream) {
	const size_t totalVoxels = data.size();

	// Get velocity block (assuming exactly one Vec3f block)
	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	const nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	// Get all float blocks (density, temperature, fuel, etc.)
	const auto floatBlocks = data.getBlocksOfType<float>();
	if (floatBlocks.empty()) {
		throw std::runtime_error("No float blocks found");
	}

	// Validate pointers
	if (!velocity) {
		throw std::runtime_error("Velocity data not found");
	}


	// Create streams and storage for float blocks
	std::vector<cudaStream_t> streams(floatBlocks.size());
	std::vector<float*> hostPointers;
	std::vector<float*> d_inputs, d_outputs;

	// Initialize resources for each float block
	for (auto& name : floatBlocks) {
		cudaStreamCreate(&streams[hostPointers.size()]);

		auto* host_ptr = data.pValues<float>(name);
		if (!host_ptr) {
			throw std::runtime_error("Block '" + name + "' not found or type mismatch");
		}
		hostPointers.push_back(host_ptr);

		float *d_in, *d_out;
		cudaMalloc(&d_in, totalVoxels * sizeof(float));
		cudaMalloc(&d_out, totalVoxels * sizeof(float));
		d_inputs.push_back(d_in);
		d_outputs.push_back(d_out);
	}


	// Allocate shared device memory
	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));

	// Copy shared data
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, mainStream);
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, mainStream);

	// Process each float block
	constexpr int blockSize = 256;
	const int numBlocks = (totalVoxels + blockSize - 1) / blockSize;
	auto gridHandle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);

	const auto gpuGrid = gridHandle.deviceGrid<nanovdb::ValueOnIndex>();


#pragma unroll
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		// Copy input data to device
		cudaMemcpyAsync(d_inputs[i], hostPointers[i], totalVoxels * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
	}
#pragma unroll
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		// Launch kernel
		advect_idx<<<numBlocks, blockSize, 0, streams[i]>>>(gpuGrid, d_coords, d_velocity, d_inputs[i], d_outputs[i], totalVoxels, dt,
		                                                    voxelSize);
	}
#pragma unroll
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		// Copy results back
		cudaMemcpyAsync(hostPointers[i], d_outputs[i], totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}

	// Synchronize and cleanup
	cudaStreamSynchronize(mainStream);
	for (auto& stream : streams) {
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}

	// Free device memory
	cudaFree(d_velocity);
	cudaFree(d_coords);
	for (size_t i = 0; i < floatBlocks.size(); ++i) {
		cudaFree(d_inputs[i]);
		cudaFree(d_outputs[i]);
	}
}


void advect_index_grid_v(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();

	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	nanovdb::Vec3f* d_velocity = nullptr;
	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	// Use synchronous copy for non-pinned memory
	cudaMemcpy(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice);

	// Allocate device memory for voxel coordinates.
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMemcpy(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice);

	// Allocate device memory for the output density.
	nanovdb::Vec3f* d_outVel = nullptr;
	cudaMalloc(&d_outVel, totalVoxels * sizeof(nanovdb::Vec3f));

	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle =
	    nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	constexpr int blockSize = 256;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;
	advect_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_outVel, totalVoxels, dt, voxelSize);

	// Make sure kernel is finished before copying back
	cudaStreamSynchronize(stream);

	// Use synchronous copy for non-pinned memory
	cudaMemcpy(velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost);

	// Free the allocated device memory.
	cudaFree(d_velocity);
	cudaFree(d_outVel);
	cudaFree(d_coords);
}


extern "C" void AdvectIndexGrid(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& stream) {
	advect_index_grid(data, dt, voxelSize, stream);
}

extern "C" void AdvectIndexGridVelocity(HNS::GridIndexedData& data, const float dt, const float voxelSize, const cudaStream_t& stream) {
	advect_index_grid_v(data, dt, voxelSize, stream);
}