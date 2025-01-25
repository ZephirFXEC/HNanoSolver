#include "PointToGrid.cuh"

extern "C" void pointToGridFloat(HNS::OpenFloatGrid& in_data, const float voxelSize, HNS::NanoFloatGrid& out_data,
                                 const cudaStream_t& stream) {
	pointToGridTemplate<float, float>(in_data, voxelSize, out_data, stream);
}

extern "C" void pointToGridVector(HNS::OpenVectorGrid& in_data, const float voxelSize, HNS::NanoVectorGrid& out_data,
                                  const cudaStream_t& stream) {
	pointToGridTemplate<openvdb::Vec3f, nanovdb::Vec3f>(in_data, voxelSize, out_data, stream);
}

extern "C" void pointToGridFloatToDevice(HNS::OpenFloatGrid& in_data, const float voxelSize,
                                         nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const cudaStream_t& stream) {
	pointToGridTemplateToDevice<float, float>(in_data, voxelSize, handle, stream);
}

extern "C" void pointToGridVectorToDevice(HNS::OpenVectorGrid& in_data, const float voxelSize,
                                          nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const cudaStream_t& stream) {
	pointToGridTemplateToDevice<openvdb::Vec3f, nanovdb::Vec3f>(in_data, voxelSize, handle, stream);
}