#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Utils.cuh"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

__global__ void divergence_idx(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coord,
                               const nanovdb::Vec3f* __restrict__ velocityData, float* __restrict__ outDivergence, const float dx,
                               const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const nanovdb::Coord coord = d_coord[tid];

	if (!velocitySampler.isDataActive(coord)) {
		return;
	}

	const nanovdb::Vec3f c = coord.asVec3s();

	const float xp = velocitySampler(c + nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float xm = velocitySampler(c - nanovdb::Vec3f(0.5f, 0.0f, 0.0f))[0];
	const float yp = velocitySampler(c + nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float ym = velocitySampler(c - nanovdb::Vec3f(0.0f, 0.5f, 0.0f))[1];
	const float zp = velocitySampler(c + nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];
	const float zm = velocitySampler(c - nanovdb::Vec3f(0.0f, 0.0f, 0.5f))[2];

	const float dixX = (xp - xm) / dx;
	const float dixY = (yp - ym) / dx;
	const float dixZ = (zp - zm) / dx;

	outDivergence[tid] = dixX + dixY + dixZ;
}

__global__ void redBlackGaussSeidelUpdate_idx(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
                                              const nanovdb::Coord* __restrict__ d_coord, const float* __restrict__ divergence,
                                              float* __restrict__ pressure, const float dx, const size_t totalVoxels, const int color,
                                              const float omega) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	const nanovdb::Coord c = d_coord[tid];
	const int i = c.x(), j = c.y(), k = c.z();

	// Skip if wrong color (reduces divergent branching)
	if ((i + j + k & 1) != color) return;

	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto divSampler = IndexSampler<float, 1>(idxSampler, divergence);
	const auto pSampler = IndexSampler<float, 1>(idxSampler, pressure);

	// Pre-compute common factors
	const float dx2 = dx * dx;
	const float inv6 = 1.0f / 6.0f;

	// Gather neighbors using vectorized loads where possible
	const float pxp1 = pSampler(nanovdb::Coord(i + 1, j, k));
	const float pxm1 = pSampler(nanovdb::Coord(i - 1, j, k));
	const float pyp1 = pSampler(nanovdb::Coord(i, j + 1, k));
	const float pym1 = pSampler(nanovdb::Coord(i, j - 1, k));
	const float pzp1 = pSampler(nanovdb::Coord(i, j, k + 1));
	const float pzm1 = pSampler(nanovdb::Coord(i, j, k - 1));

	const float divVal = divSampler(c);
	const float pOld = pSampler(c);

	// Optimize arithmetic operations
	const float pGS = ((pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1) - divVal * dx2) * inv6;
	pressure[tid] = pOld + omega * (pGS - pOld);
}


__global__ void subtractPressureGradient_idx(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid,
                                             const nanovdb::Coord* __restrict__ d_coords, const size_t totalVoxels,
                                             const nanovdb::Vec3f* __restrict__ velocity,  // velocity at faces
                                             const float* __restrict__ pressure,           // pressure at cell centers
                                             nanovdb::Vec3f* __restrict__ out, float voxelSize) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels) return;

	// Accessors / Samplers
	const IndexOffsetSampler<0> idxSampler(*domainGrid);
	const auto pressureSampler = IndexSampler<float, 1>(idxSampler, pressure);
	const auto velSampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocity);


	const float dx = voxelSize;

	// The cell center coordinate
	const nanovdb::Vec3f c = d_coords[tid].asVec3s();
	const nanovdb::Vec3f vel = sampleMACVelocity_idx(velSampler, c);
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

	out[tid] = v;
}

void pressure_projection_idx(HNS::GridIndexedData& data, const size_t iteration, const float voxelSize, const cudaStream_t& stream) {
	const size_t totalVoxels = data.size();
	constexpr int blockSize = 256;
	int numBlocks = (totalVoxels + blockSize - 1) / blockSize;

	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity)");
	}

	nanovdb::Vec3f* velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(vec3fBlocks[0]));

	nanovdb::Vec3f* d_velocity = nullptr;
	nanovdb::Coord* d_coords = nullptr;
	float* d_divergence = nullptr;
	float* d_pressure = nullptr;

	cudaMalloc(&d_velocity, totalVoxels * sizeof(nanovdb::Vec3f));
	cudaMalloc(&d_coords, totalVoxels * sizeof(nanovdb::Coord));
	cudaMalloc(&d_divergence, totalVoxels * sizeof(float));
	cudaMalloc(&d_pressure, totalVoxels * sizeof(float));

	// Use async memory operations with stream
	cudaMemcpyAsync(d_velocity, velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_coords, data.pCoords(), totalVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, stream);
	cudaMemsetAsync(d_divergence, 0, totalVoxels * sizeof(float), stream);
	cudaMemsetAsync(d_pressure, 0, totalVoxels * sizeof(float), stream);

	// Create grid handle and get device grid pointer
	auto handle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);

	const auto gpuGrid = handle.deviceGrid<nanovdb::ValueOnIndex>();

	divergence_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_velocity, d_divergence, voxelSize, totalVoxels);

	// Red-black Gauss-Seidel iterations
	constexpr float omega = 1.9f;  // SOR relaxation parameter
	for (int iter = 0; iter < iteration; ++iter) {
		redBlackGaussSeidelUpdate_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize,
		                                                                   totalVoxels, 0, omega);
		redBlackGaussSeidelUpdate_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, d_divergence, d_pressure, voxelSize,
		                                                                   totalVoxels, 1, omega);
	}

	// Apply pressure gradient
	subtractPressureGradient_idx<<<numBlocks, blockSize, 0, stream>>>(gpuGrid, d_coords, totalVoxels, d_velocity, d_pressure, d_velocity,
	                                                                  voxelSize);

	// Copy result back asynchronously
	cudaMemcpyAsync(velocity, d_velocity, totalVoxels * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, stream);

	// Clean up resources
	cudaStreamSynchronize(stream);

	cudaFree(d_velocity);
	cudaFree(d_coords);
	cudaFree(d_divergence);
	cudaFree(d_pressure);
}


extern "C" void Divergence_idx(HNS::GridIndexedData& data, const size_t iterations, const float voxelSize, const cudaStream_t& stream) {
	pressure_projection_idx(data, iterations, voxelSize, stream);
}