#pragma once

#include "Utils.cuh"
#include "nanovdb/NanoVDB.h"

#define COMBUSTION_RATE 0.1f         // Rate at which fuel burns
#define HEAT_RELEASE 10.0f           // Heat released per unit of fuel burned
#define BUOYANCY_STRENGTH 1.0f       // Strength of buoyant force
#define AMBIENT_TEMP 23.0f           // Ambient temperature in Celsius
#define TEMPERATURE_DIFFUSION 0.02f  // Temperature diffusion coefficient
#define FUEL_DIFFUSION 0.01f         // Fuel diffusion coefficient
#define IGNITION_TEMP 150.0f         // Temperature required for combustion

struct CombustionParams {
	float combustionRate = COMBUSTION_RATE;
	float heatRelease = HEAT_RELEASE;
	float buoyancyStrength = BUOYANCY_STRENGTH;
	float ambientTemp = AMBIENT_TEMP;
	float temperatureDiffusion = TEMPERATURE_DIFFUSION;
	float fuelDiffusion = FUEL_DIFFUSION;
	float ignitionTemp = IGNITION_TEMP;
};

__global__ void advect_scalar(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              const float* __restrict__ inData, float* __restrict__ outData, size_t totalVoxels, float dt,
                              float inv_voxelSize);

__global__ void advect_vector(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              nanovdb::Vec3f* __restrict__ outVelocity, size_t totalVoxels, float dt, float inv_voxelSize);

__global__ void divergence_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Vec3f* velocityData,
                               float* outDivergence, float inv_dx, int numLeaves);

__global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                           const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                           float* __restrict__ outDivergence, float inv_dx, size_t totalVoxels);


__global__ void redBlackGaussSeidelUpdate_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                              const float* __restrict__ d_divergence, float* __restrict__ d_pressure, float voxelSize,
                                              int numLeaves, int color, float omega);

__global__ void redBlackGaussSeidelUpdate(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                          const nanovdb::Coord* __restrict__ d_coord, const float* __restrict__ divergence,
                                          float* __restrict__ pressure, float dx, size_t totalVoxels, int color, float omega);

__global__ void subtractPressureGradient_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                             const nanovdb::Vec3f* __restrict__ velocity, const float* __restrict__ pressure,
                                             nanovdb::Vec3f* __restrict__ out, float inv_voxelSize, size_t numLeaves);


__global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                         const nanovdb::Coord* __restrict__ d_coords, size_t totalVoxels,
                                         const nanovdb::Vec3f* __restrict__ velocity, const float* __restrict__ pressure,
                                         nanovdb::Vec3f* __restrict__ out, float inv_voxelSize);

__global__ void temperature_buoyancy(const nanovdb::Vec3f* velocityData, const float* tempData, nanovdb::Vec3f* outVel, const float dt,
                                     float ambient_temp, float buoyancy_strength, size_t totalVoxels);

__global__ void combustion(const float* fuelData, const float* tempData, float* outFuel, float* outTemp, const float dt,
                           float ignition_temp, float combustion_rate, float heat_release, size_t totalVoxels);

__global__ void diffusion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                          const float* tempData, const float* fuelData, float* outTemp, float* outFuel, const float dt, float temp_diff,
                          float fuel_diff, float ambient_temp, size_t totalVoxels);