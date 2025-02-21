#pragma once

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "BrickMap.cuh"

#include <nanovdb/math/Math.h>
#include <nanovdb/util/cuda/Util.h>


class VolumeTexture {
   public:
	VolumeTexture() = default;
	~VolumeTexture() {
		if (cudaResource_) {
			cudaGraphicsUnregisterResource(cudaResource_);
		}
		if (tex_) {
			glDeleteTextures(1, &tex_);
		}
	}

	void create(const BrickMap& brickMap);
	void update(const BrickMap& brickMap);

	GLuint texture() const { return tex_; }

   private:
	GLuint tex_ = 0;
	cudaGraphicsResource* cudaResource_ = nullptr;
	uint32_t dim_[3] = {0};
};
