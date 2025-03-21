#pragma once

#include "../Utils/Stencils.hpp"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/math/SampleFromVoxels.h"

template <typename T>
__device__ T lerp(T v0, T v1, T t) {
	return fma(t, v1, fma(-t, v0, v0));
}

inline size_t blocksPerGrid(const size_t numItems, const size_t threadsPerBlock) {
	NANOVDB_ASSERT(numItems > 0 && threadsPerBlock >= 32 && threadsPerBlock % 32 == 0);
	return (numItems + threadsPerBlock - 1) / threadsPerBlock;
}

__device__ __forceinline__ float computeError(const float original, const float backward) {
	return __fmul_rn(0.5f, __fsub_rn(original, backward));
}

__device__ __forceinline__ nanovdb::Vec3f computeError(const nanovdb::Vec3f& original, const nanovdb::Vec3f& backward) {
	return nanovdb::Vec3f(computeError(original[0], backward[0]), computeError(original[1], backward[1]),
	                      computeError(original[2], backward[2]));
}

__device__ __forceinline__ float absValue(const float x) { return fabsf(x); }

__device__ __forceinline__ nanovdb::Vec3f absValue(const nanovdb::Vec3f& v) {
	return nanovdb::Vec3f(fabsf(v[0]), fabsf(v[1]), fabsf(v[2]));
}

__device__ __forceinline__ float computeMaxCorrection(const float forward, const float original) {
	return __fmul_rn(0.5f, fabsf(__fsub_rn(forward, original)));
}
__device__ __forceinline__ nanovdb::Vec3f computeMaxCorrection(const nanovdb::Vec3f& forward, const nanovdb::Vec3f& original) {
	return nanovdb::Vec3f(computeMaxCorrection(forward[0], original[0]), computeMaxCorrection(forward[1], original[1]),
	                      computeMaxCorrection(forward[2], original[2]));
}


__device__ __forceinline__ float clampValue(const float x, const float minVal, const float maxVal) {
	return fminf(fmaxf(x, minVal), maxVal);
}

__device__ __forceinline__ nanovdb::Vec3f clampValue(const nanovdb::Vec3f& x, const nanovdb::Vec3f& minVal, const nanovdb::Vec3f& maxVal) {
	return nanovdb::Vec3f(fminf(fmaxf(x[0], minVal[0]), maxVal[0]), fminf(fmaxf(x[1], minVal[1]), maxVal[1]),
	                      fminf(fmaxf(x[2], minVal[2]), maxVal[2]));
}

__device__ __forceinline__ float lerp(const float a, const float b, const float t) { return a + t * (b - a); }

__device__ __forceinline__ nanovdb::Vec3f lerp(const nanovdb::Vec3f& a, const nanovdb::Vec3f& b, const float t) { return a + (b - a) * t; }

__device__ __forceinline__ float enforceNonNegative(const float x) { return fmaxf(0.0f, x); }

__device__ __forceinline__ int clampCoord(const int c, const int minC, const int maxC) {
	if (c < minC) return minC;
	if (c > maxC) return maxC;
	return c;
}

inline __device__ nanovdb::Vec3f FaceVelocity(const IndexSampler<nanovdb::Vec3f, 1>& velSampler, const nanovdb::Coord& pos) {
	const nanovdb::Vec3f center = velSampler(pos);
	const float u = 0.5f * (velSampler(pos - nanovdb::Coord(1, 0, 0))[0] + center[0]);
	const float v = 0.5f * (velSampler(pos - nanovdb::Coord(0, 1, 0))[1] + center[1]);
	const float w = 0.5f * (velSampler(pos - nanovdb::Coord(0, 0, 1))[2] + center[2]);
	return {u, v, w};
}

inline __device__ nanovdb::Vec3f MACToFaceCentered(const IndexSampler<nanovdb::Vec3f, 1>& velSampler, const nanovdb::Coord& pos) {
	nanovdb::Vec3f center = velSampler(pos);

	// Left and right face velocity
	const float xm = 0.5f * (velSampler(pos - nanovdb::Coord(1, 0, 0))[0] + center[0]);
	const float xp = 0.5f * (center[0] + velSampler(pos + nanovdb::Coord(1, 0, 0))[0]);

	// Top and bottom face velocity
	const float ym = 0.5f * (velSampler(pos - nanovdb::Coord(0, 1, 0))[1] + center[1]);
	const float yp = 0.5f * (center[1] + velSampler(pos + nanovdb::Coord(0, 1, 0))[1]);

	// Front and back face velocity
	const float zm = 0.5f * (velSampler(pos - nanovdb::Coord(0, 0, 1))[2] + center[2]);
	const float zp = 0.5f * (center[2] + velSampler(pos + nanovdb::Coord(0, 0, 1))[2]);

	return nanovdb::Vec3f(0.5f * (xm + xp), 0.5f * (ym + yp), 0.5f * (zm + zp));
}

inline __device__ nanovdb::Vec3f MACToFaceCentered(const IndexSampler<nanovdb::Vec3f, 1>& velSampler, const nanovdb::Vec3f& pos) {
	nanovdb::Vec3f result;

	// Precompute the base positions for each component
	const nanovdb::Vec3f adj_pos_x = pos - nanovdb::Vec3f(0.5f, 0.0f, 0.0f);
	const nanovdb::Vec3f adj_pos_y = pos - nanovdb::Vec3f(0.0f, 0.5f, 0.0f);
	const nanovdb::Vec3f adj_pos_z = pos - nanovdb::Vec3f(0.0f, 0.0f, 0.5f);

	const nanovdb::Coord base_x = adj_pos_x.floor();
	const nanovdb::Coord base_y = adj_pos_y.floor();
	const nanovdb::Coord base_z = adj_pos_z.floor();

	const nanovdb::Vec3f frac_x = adj_pos_x - nanovdb::Vec3f(base_x);
	const nanovdb::Vec3f frac_y = adj_pos_y - nanovdb::Vec3f(base_y);
	const nanovdb::Vec3f frac_z = adj_pos_z - nanovdb::Vec3f(base_z);

	// Cache sampled velocities to avoid redundant lookups
	nanovdb::Vec3f velocities[27];  // 3x3x3 grid to cache all needed samples

	// Sample velocities for all needed positions
	for (int z = 0; z <= 2; ++z) {
		for (int y = 0; y <= 2; ++y) {
			for (int x = 0; x <= 2; ++x) {
				// Base position for the current component
				nanovdb::Coord baseCoord = base_x + nanovdb::Coord(x - 1, y - 1, z - 1);
				int idx = x + 3 * y + 9 * z;
				velocities[idx] = velSampler(baseCoord);
			}
		}
	}

	// Helper function for trilinear interpolation
	auto trilinear = [](const float values[8], const nanovdb::Vec3f& frac) {
		const float v00 = lerp(values[0], values[1], frac[0]);
		const float v01 = lerp(values[2], values[3], frac[0]);
		const float v0 = lerp(v00, v01, frac[1]);

		const float v10 = lerp(values[4], values[5], frac[0]);
		const float v11 = lerp(values[6], values[7], frac[0]);
		const float v1 = lerp(v10, v11, frac[1]);

		return lerp(v0, v1, frac[2]);
	};

	// X-component interpolation (staggered at x-faces)
	{
		float vx[8];
		for (int z = 0; z <= 1; ++z) {
			for (int y = 0; y <= 1; ++y) {
				for (int x = 0; x <= 1; ++x) {
					// Calculate indices into our cached velocities
					const int idx1 = (x + 0) + 3 * (y + 1) + 9 * (z + 1);  // base_x + (x,y,z)
					const int idx2 = (x + 1) + 3 * (y + 1) + 9 * (z + 1);  // base_x + (x+1,y,z)
					vx[x + 2 * y + 4 * z] = 0.5f * (velocities[idx1][0] + velocities[idx2][0]);
				}
			}
		}
		result[0] = trilinear(vx, frac_x);
	}

	// Y-component interpolation (staggered at y-faces)
	{
		float vy[8];
		for (int z = 0; z <= 1; ++z) {
			for (int y = 0; y <= 1; ++y) {
				for (int x = 0; x <= 1; ++x) {
					// Calculate indices into our cached velocities
					int idx1 = (x + 1) + 3 * (y + 0) + 9 * (z + 1);  // base_y + (x,y,z)
					int idx2 = (x + 1) + 3 * (y + 1) + 9 * (z + 1);  // base_y + (x,y+1,z)
					vy[x + 2 * y + 4 * z] = 0.5f * (velocities[idx1][1] + velocities[idx2][1]);
				}
			}
		}
		result[1] = trilinear(vy, frac_y);
	}

	// Z-component interpolation (staggered at z-faces)
	{
		float vz[8];
		for (int z = 0; z <= 1; ++z) {
			for (int y = 0; y <= 1; ++y) {
				for (int x = 0; x <= 1; ++x) {
					// Calculate indices into our cached velocities
					int idx1 = (x + 1) + 3 * (y + 1) + 9 * (z + 0);  // base_z + (x,y,z)
					int idx2 = (x + 1) + 3 * (y + 1) + 9 * (z + 1);  // base_z + (x,y,z+1)
					vz[x + 2 * y + 4 * z] = 0.5f * (velocities[idx1][2] + velocities[idx2][2]);
				}
			}
		}
		result[2] = trilinear(vz, frac_z);
	}

	return result;
}

template <typename VelocitySampler>
__device__ nanovdb::Vec3f rk4_integrate(const VelocitySampler& sampler, nanovdb::Vec3f start_pos, float h) {
	const nanovdb::Vec3f k1 = sampler(start_pos) * h;
	const nanovdb::Vec3f k2 = sampler(start_pos + 0.5f * k1) * h;
	const nanovdb::Vec3f k3 = sampler(start_pos + 0.5f * k2) * h;
	const nanovdb::Vec3f k4 = sampler(start_pos + k3) * h;

	nanovdb::Vec3f result = start_pos;
	result[0] = __fmaf_rn(0.16667f, k1[0] + k4[0] + 2.0f * (k2[0] + k3[0]), start_pos[0]);
	result[1] = __fmaf_rn(0.16667f, k1[1] + k4[1] + 2.0f * (k2[1] + k3[1]), start_pos[1]);
	result[2] = __fmaf_rn(0.16667f, k1[2] + k4[2] + 2.0f * (k2[2] + k3[2]), start_pos[2]);
	return result;
}

template <typename VelocitySampler>
__device__ nanovdb::Vec3f rk3_integrate(const VelocitySampler& sampler, nanovdb::Vec3f start_pos, float h) {
	const nanovdb::Vec3f k1 = sampler(start_pos) * h;
	const nanovdb::Vec3f k2 = sampler(start_pos + 0.5f * k1) * h;
	const nanovdb::Vec3f k3 = sampler(start_pos - k1 + 2.0f * k2) * h;

	nanovdb::Vec3f result = start_pos;
	result[0] = __fmaf_rn(0.33333f, k1[0] + 3.0f * k2[0] + k3[0], start_pos[0]);
	result[1] = __fmaf_rn(0.33333f, k1[1] + 3.0f * k2[1] + k3[1], start_pos[1]);
	result[2] = __fmaf_rn(0.33333f, k1[2] + 3.0f * k2[2] + k3[2], start_pos[2]);
	return result;
}


class ScopedTimerGPU {
   public:
	explicit ScopedTimerGPU(const std::string& name) : name_(name), bytes(0), voxels(0) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	ScopedTimerGPU(const std::string& name, const uint16_t bytes, const size_t voxels) : name_(name), bytes(bytes), voxels(voxels) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	~ScopedTimerGPU() {
		float elapsed;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);

		printf("-- %s -- \n", name_.c_str());
		printf("Bandwidth: %f GB/s\n", voxels * (bytes / 1e9) / (elapsed / 1e3));
		printf("Time: %f ms\n", elapsed);
	}

   private:
	const std::string name_{};
	const uint16_t bytes;
	const size_t voxels;
	cudaEvent_t start{};
	cudaEvent_t stop{};
};