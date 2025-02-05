#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cuda/std/cmath>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "PointToGrid.cuh"

__global__ void advect(const CudaResources<nanovdb::Coord, float, true> resources, const size_t npoints, const float dt,
                       const float voxelSize, const nanovdb::Vec3fGrid* __restrict__ vel_grid,
                       const nanovdb::FloatGrid* __restrict__ d_grid) {
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

__global__ void advect(const CudaResources<nanovdb::Coord, nanovdb::Vec3f, true> resources, const size_t npoints, const float dt,
                       const float voxelSize, const nanovdb::Vec3fGrid* __restrict__ vel_grid,
                       const nanovdb::Vec3fGrid* __restrict__ d_grid) {
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


__global__ void advect_idx(
	const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
	const nanovdb::Coord* __restrict__ coords,
	const nanovdb::Vec3f* __restrict__ velocityData,
	const float* __restrict__ densityData,
	const float* __restrict__ temperatureData,
	const float* __restrict__ fuelData,
	float* __restrict__ outDensity,
	float* __restrict__ outTemperature,
	float* __restrict__ outFuel,
	const size_t totalVoxels,
	const float dt,
	const float voxelSize)
{
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);

	const auto densitySampler = IndexSampler<float, 1>(idxSampler, densityData);
	const auto temperatureSampler = IndexSampler<float, 1>(idxSampler, temperatureData);
	const auto fuelSampler = IndexSampler<float, 1>(idxSampler, fuelData);

	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f velocity = velocitySampler(coord);

	const nanovdb::Vec3f displacedPos = coord.asVec3s() - velocity * dt / voxelSize;

	outDensity[idx] = densitySampler(displacedPos);
	outTemperature[idx] = temperatureSampler(displacedPos);
	outFuel[idx] = fuelSampler(displacedPos);
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

	const IndexOffsetSampler<0> idxSampler(*domainGrid);

	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f velocity = velocitySampler(coord);

	const nanovdb::Vec3f displacedPos = coord.asVec3s() - velocity * dt / voxelSize;

	outVelocity[idx] = velocitySampler(displacedPos);
}


void advect_index_grid(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid, HNS::GridIndexedData& data, const float dt,
                       const float voxelSize, const cudaStream_t stream) {
	const size_t totalVoxels = data.size();

	const nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>("vel"));
	auto* density = data.pValues<float>("density");
	auto* temperature = data.pValues<float>("temperature");
	auto* fuel = data.pValues<float>("fuel");

	if (!velocity || !density || !temperature || !fuel) {
		throw std::runtime_error("Density data not found in the grid.");
	}

	// Allocate device memory for velocity.

	nanovdb::Vec3f* d_velocity = nullptr;
	float* d_density = nullptr;
	float* d_temperature = nullptr;
	float* d_fuel = nullptr;

	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_density, totalVoxels * sizeof(float));
	cudaMalloc(&d_temperature, totalVoxels * sizeof(float));
	cudaMalloc(&d_fuel, totalVoxels * sizeof(float));

	cudaMemcpyAsync(d_density, density, totalVoxels * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_temperature, temperature, totalVoxels * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_fuel, fuel, totalVoxels * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);

	// Allocate device memory for voxel coordinates.
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);

	// Allocate device memory for the output density.
	float* d_outDensity = nullptr;
	float* d_outTemperature = nullptr;
	float* d_outFuel = nullptr;

	cudaMalloc(&d_outDensity, totalVoxels * sizeof(float));
	cudaMalloc(&d_outTemperature, totalVoxels * sizeof(float));
	cudaMalloc(&d_outFuel, totalVoxels * sizeof(float));

	constexpr int blockSize = 256;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;
	advect_idx<<<numBlocks, blockSize, 0, stream>>>(indexGrid, d_coords, d_velocity, d_density, d_temperature, d_fuel, d_outDensity,
	                                                d_outTemperature, d_outFuel, totalVoxels, dt, voxelSize);


	cudaMemcpyAsync(density, d_outDensity, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(temperature, d_outTemperature, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(fuel, d_outFuel, totalVoxels * sizeof(float), cudaMemcpyDeviceToHost, stream);

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


void advect_index_grid_v(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid, HNS::GridIndexedData& data, const float dt,
                       const float voxelSize, const cudaStream_t stream) {
	const size_t totalVoxels = data.size();

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>("vel"));

	nanovdb::Vec3f* d_velocity = nullptr;
	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);

	// Allocate device memory for voxel coordinates.
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);

	// Allocate device memory for the output density.
	nanovdb::Vec3f* d_outVel = nullptr;
	cudaMalloc(&d_outVel, totalVoxels * sizeof(nanovdb::Vec3f));


	constexpr int blockSize = 256;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;
	advect_idx<<<numBlocks, blockSize, 0, stream>>>(indexGrid, d_coords, d_velocity, d_outVel, totalVoxels, dt, voxelSize);


	cudaMemcpyAsync(velocity, d_outVel, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, stream);

	// Free the allocated device memory.
	cudaFree(d_velocity);
	cudaFree(d_outVel);
	cudaFree(d_coords);
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

extern "C" void AdvectIndexGrid(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid, HNS::GridIndexedData& data,
                                const float dt, const float voxelSize, const cudaStream_t stream) {
	advect_index_grid(indexGrid, data, dt, voxelSize, stream);
}

extern "C" void AdvectIndexGridVelocity(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid, HNS::GridIndexedData& data, const float dt,
				const float voxelSize, const cudaStream_t stream) {
	advect_index_grid_v(indexGrid, data, dt, voxelSize, stream);
}