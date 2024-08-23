#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/GridHandle.cuh>

extern "C"  void scaleActiveVoxels(nanovdb::FloatGrid *grid_d, const uint64_t leafCount, float scale)
{
	auto kernel = [grid_d, scale] __device__(const uint64_t n) {
		auto *leaf_d =
		    grid_d->tree().getFirstNode<0>() + (n >> 9);  // this only works if grid->isSequential<0>() == true
		const int i = n & 511;
		const float v = scale * leaf_d->getValue(i);
		if (leaf_d->isActive(i)) {
			leaf_d->setValueOnly(i, v);  // only possible execution divergence
		}
	};

	const thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
	thrust::for_each(iter, iter + 512*leafCount, kernel);
}