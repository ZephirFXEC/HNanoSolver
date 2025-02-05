#include <nanovdb/NanoVDB.h>

#include <cuda/std/cmath>
#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"

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


extern "C" void AdvectIndexGrid(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid, HNS::GridIndexedData& data,
                                const float dt, const float voxelSize, const cudaStream_t stream) {
	advect_index_grid(indexGrid, data, dt, voxelSize, stream);
}

extern "C" void AdvectIndexGridVelocity(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* indexGrid, HNS::GridIndexedData& data, const float dt,
				const float voxelSize, const cudaStream_t stream) {
	advect_index_grid_v(indexGrid, data, dt, voxelSize, stream);
}