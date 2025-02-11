#include <nanovdb/NanoVDB.h>

#include <cuda/std/cmath>
#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include <openvdb/Types.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include "Utils.cuh"

__global__ void advect_idx(
	const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
	const nanovdb::Coord* __restrict__ coords,
	const nanovdb::Vec3f* __restrict__ velocityData,
	const float* __restrict__ inData,
	float* __restrict__ outData,
	const size_t totalVoxels,
	const float dt,
	const float voxelSize)
{
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


__global__ void advect_idx(
	const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
	const nanovdb::Coord* __restrict__ coords,
	const nanovdb::Vec3f* __restrict__ velocityData,
	nanovdb::Vec3f* __restrict__ outVelocity,
	const size_t totalVoxels,
	const float dt,
	const float voxelSize)
{
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


void advect_index_grid(HNS::GridIndexedData& data, const float dt,
                       const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();

	const nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>("vel"));
	auto* density = data.pValues<float>("density");
	auto* temperature = data.pValues<float>("temperature");
	auto* fuel = data.pValues<float>("fuel");

	if (!velocity || !density || !temperature || !fuel) {
		throw std::runtime_error("Density data not found in the grid.");
	}

	// Allocate device memory.
	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;

	float* d_density = nullptr;
	float* d_temperature = nullptr;
	float* d_fuel = nullptr;

	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMalloc(&d_density, totalVoxels * sizeof(float));
	cudaMalloc(&d_temperature, totalVoxels * sizeof(float));
	cudaMalloc(&d_fuel, totalVoxels * sizeof(float));

	cudaMemcpy(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice);
	cudaMemcpy(d_density, density, totalVoxels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_temperature, temperature, totalVoxels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fuel, fuel, totalVoxels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice);

	// Allocate device memory for the output density.
	float* d_outDensity = nullptr;
	float* d_outTemperature = nullptr;
	float* d_outFuel = nullptr;

	cudaMalloc(&d_outDensity, totalVoxels * sizeof(float));
	cudaMalloc(&d_outTemperature, totalVoxels * sizeof(float));
	cudaMalloc(&d_outFuel, totalVoxels * sizeof(float));

	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle =
	nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);


	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	constexpr int blockSize = 512;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;

	advect_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_density, d_outDensity, totalVoxels, dt, voxelSize);
	advect_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_temperature, d_outTemperature, totalVoxels, dt, voxelSize);
	advect_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_fuel, d_outFuel, totalVoxels, dt, voxelSize);

	cudaStreamSynchronize(stream);

	cudaMemcpy(density, d_outDensity, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(temperature, d_outTemperature, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(fuel, d_outFuel, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost);

	// Free the allocated device memory.
	cudaFree(d_velocity);
	cudaFree(d_density);
	cudaFree(d_coords);
	cudaFree(d_outDensity);
	cudaFree(d_temperature);
	cudaFree(d_outTemperature);
	cudaFree(d_fuel);
	cudaFree(d_outFuel);
}


void advect_index_grid_v(HNS::GridIndexedData& data, const float dt,
                       const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>("vel"));

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
	cudaDeviceSynchronize();

	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle =
	nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);
	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();
	cudaDeviceSynchronize();

	constexpr int blockSize = 256;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;
	advect_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_outVel, totalVoxels, dt, voxelSize);

	// Make sure kernel is finished before copying back
	cudaDeviceSynchronize();

	// Use synchronous copy for non-pinned memory
	cudaMemcpy(velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost);

	// Free the allocated device memory.
	cudaFree(d_velocity);
	cudaFree(d_outVel);
	cudaFree(d_coords);
}


extern "C" void AdvectIndexGrid(HNS::GridIndexedData& data,
                                const float dt, const float voxelSize, const cudaStream_t& stream) {
	advect_index_grid(data, dt, voxelSize, stream);
}

extern "C" void AdvectIndexGridVelocity(HNS::GridIndexedData& data, const float dt,
				const float voxelSize, const cudaStream_t& stream) {
	advect_index_grid_v(data, dt, voxelSize, stream);
}