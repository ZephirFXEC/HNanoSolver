#include "VolumeTex.cuh"

__global__ void copyBricksToTexture(const uint32_t* grid, const Voxel* voxelData, cudaSurfaceObject_t surf, uint32_t gridX, uint32_t gridY,
                                    uint32_t brickSize) {
	const uint32_t brickIdx = blockIdx.z * gridY * gridX + blockIdx.y * gridX + blockIdx.x;
	if (grid[brickIdx] == BrickConfig::INACTIVE) return;

	const uint32_t voxelX = threadIdx.x;
	const uint32_t voxelY = threadIdx.y;
	const uint32_t voxelZ = threadIdx.z;

	const uint32_t globalX = blockIdx.x * brickSize + voxelX;
	const uint32_t globalY = blockIdx.y * brickSize + voxelY;
	const uint32_t globalZ = blockIdx.z * brickSize + voxelZ;

	const Voxel v =
	    voxelData[grid[brickIdx] * (brickSize * brickSize * brickSize) + voxelX + voxelY * brickSize + voxelZ * brickSize * brickSize];

	surf3Dwrite(v.density, surf, globalX * sizeof(float), globalY, globalZ);
}

void VolumeTexture::create(const BrickMap& brickMap) {
	const auto dim = brickMap.getDimensions();
	dim_[0] = dim[0] * BrickConfig::BRICK_SIZE;
	dim_[1] = dim[1] * BrickConfig::BRICK_SIZE;
	dim_[2] = dim[2] * BrickConfig::BRICK_SIZE;

	// Create OpenGL texture
	glGenTextures(1, &tex_);
	glBindTexture(GL_TEXTURE_3D, tex_);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, dim_[0], dim_[1], dim_[2], 0, GL_RED, GL_FLOAT, nullptr);

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Register with CUDA
	cudaGraphicsGLRegisterImage(&cudaResource_, tex_, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}


void VolumeTexture::update(const BrickMap& brickMap) {
	cudaArray* array;
	cudaGraphicsMapResources(1, &cudaResource_, nullptr);
	cudaGraphicsSubResourceGetMappedArray(&array, cudaResource_, 0, 0);

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = array;

	cudaSurfaceObject_t surf;
	cudaCreateSurfaceObject(&surf, &resDesc);

	const auto dim = brickMap.getDimensions();
	constexpr uint32_t brickSize = BrickConfig::BRICK_SIZE;

	dim3 block(brickSize, brickSize, brickSize);
	dim3 grid(dim[0], dim[1], dim[2]);

	copyBricksToTexture<<<grid, block>>>(brickMap.getDeviceGrid(), brickMap.getPool()->getDeviceVoxelData(), surf, dim[0], dim[1],
	                                     brickSize);

	cudaDestroySurfaceObject(surf);
	cudaGraphicsUnmapResources(1, &cudaResource_, nullptr);
}

extern "C" {
void Create(VolumeTexture* volumeTexture, const BrickMap* brickMap) { volumeTexture->create(*brickMap); }
void Update(VolumeTexture* volumeTexture, const BrickMap* brickMap) { volumeTexture->update(*brickMap); }
}