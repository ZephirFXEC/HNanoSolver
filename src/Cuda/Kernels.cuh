#pragma once

#include "Utils.cuh"
#include "nanovdb/NanoVDB.h"

struct CombustionParams {
	float expansionRate;
	float temperatureRelease;
	float buoyancyStrength;
	float ambientTemp;
	float vorticityScale;
	float factorScale;
};

__global__ void advect_scalar(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              const float* __restrict__ inData, float* __restrict__ outData, const float* __restrict__ collisionSDF,
                              const bool hasCollision, size_t totalVoxels, float dt, float inv_voxelSize);

__global__ void advect_scalars(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                               const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                               float** __restrict__ inDataArrays, float** __restrict__ outDataArrays, int numScalars,
                               const float* __restrict__ collisionSDF, const bool hasCollision, size_t totalVoxels, float dt,
                               float inv_voxelSize);

__global__ void advect_vector(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              nanovdb::Vec3f* __restrict__ outVelocity, const float* __restrict__ collisionSDF, const bool hasCollision,
                              size_t totalVoxels, float dt, float inv_voxelSize);

__global__ void divergence_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Vec3f* velocityData,
                               float* outDivergence, float inv_dx, int numLeaves);

__global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                           const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                           float* __restrict__ outDivergence, float inv_dx, size_t totalVoxels);

__global__ void restrict_to_4x4x4(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                  const float* inData, float* outData, size_t totalVoxels);
__global__ void prolongate(const float* __restrict__ coarse, float* __restrict__ fine, nanovdb::Coord coarse_dims,
                           nanovdb::Coord fine_dims);

__global__ void update_pressure(size_t totalVoxels, const float* pressure, const float* correction);

__global__ void restrict_to_2x2x2(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                  const float* inData, float* outData, size_t totalVoxels);

__global__ void compute_residual(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                 const float* pressure, const float* divergence, float* residual, float dx, size_t totalVoxels);


__global__ void redBlackGaussSeidelUpdate_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const float* divergence,
                                              float* pressure, float dx, size_t totalVoxels, int color, float omega);

__global__ void redBlackGaussSeidelUpdate(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                          const nanovdb::Coord* __restrict__ d_coord, const float* __restrict__ divergence,
                                          float* __restrict__ pressure, float dx, size_t totalVoxels, int color, float omega);

__global__ void subtractPressureGradient_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                             const nanovdb::Vec3f* __restrict__ velocity, const float* __restrict__ pressure,
                                             nanovdb::Vec3f* __restrict__ out, float inv_voxelSize, size_t numLeaves);


__global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                         size_t totalVoxels, const nanovdb::Vec3f* velocity, const float* pressure, nanovdb::Vec3f* out,
                                         const float* __restrict__ collisionSDF, const bool hasCollision, float inv_voxelSize);

__global__ void temperature_buoyancy(const nanovdb::Vec3f* velocityData, const float* tempData, nanovdb::Vec3f* outVel, float dt,
                                     float ambient_temp, float buoyancy_strength, size_t totalVoxels);

__global__ void combustion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                           const float* __restrict__ fuelData, const float* __restrict__ tempData, float* __restrict__ outFuel,
                           float* __restrict__ outTemp, const float dt, float ignition_temp, float combustion_rate, float heat_release,
                           size_t totalVoxels);

__global__ void diffusion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                          const float* tempData, const float* fuelData, float* outTemp, float* outFuel, const float dt, float temp_diff,
                          float fuel_diff, float ambient_temp, size_t totalVoxels);


__global__ void combustion_oxygen(const float* fuelData, const float* wasteData, const float* temperatureData, float* divergenceData,
                                  const float* flameData, float* outFuel, float* outWaste, float* outTemperature, float* outFlame,
                                  float temp_gain, float expansion, size_t totalVoxels);

__global__ void vorticityConfinement(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                     const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                                     nanovdb::Vec3f* __restrict__ outForce, float dt, float inv_dx, float confinementScale,
                                     float factorScale, size_t totalVoxels);

// New functions for collision handling
template <typename Vec3T>
__device__ float sampleSDF(const float* sdfData, const Vec3T& coord, const IndexSampler<float, 1>& sampler);

template <typename Vec3T>
__device__ bool isInCollision(const float* sdfData, const Vec3T& pos, const IndexSampler<float, 1>& sampler, float threshold = 0.0f);

__device__ nanovdb::Vec3f applyNoSlipBoundary(const nanovdb::Vec3f& velocity, const nanovdb::Vec3f& normal);

template <typename Vec3T>
__device__ nanovdb::Vec3f getSDFNormal(const float* sdfData, Vec3T& pos, const IndexSampler<float, 1>& sampler, float epsilon = 0.015f);

__global__ void enforceCollisionBoundaries(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                           const nanovdb::Coord* __restrict__ coords, nanovdb::Vec3f* __restrict__ velocityData,
                                           const float* __restrict__ collisionSDF, const float voxelSize, size_t totalVoxels);

template <typename Vec3T>
__device__ nanovdb::Vec3f gradientSDF(const float* sdfData, const Vec3T& coord, const IndexSampler<float, 1>& sampler, float inv_voxelSize);