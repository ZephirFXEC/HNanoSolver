//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <cuda_runtime.h>
#include <nanovdb/NanoVDB.h>
#include <openvdb/openvdb.h>

#include <memory>
#include <stdexcept>
namespace HNS {

enum class AllocationType { Standard, Aligned, CudaPinned };

template <typename T>
struct MemoryBlock {
	std::unique_ptr<T[]> ptr{nullptr};
	AllocationType allocType = AllocationType::Standard;
	size_t size = 0;

	MemoryBlock() = default;

	void allocateCudaPinned(const size_t numElements) {
		clear();
		void* temp;
		if (cudaMallocHost(&temp, numElements * sizeof(T)) != cudaSuccess) {
			throw std::runtime_error("Failed to allocate pinned memory");
		}
		ptr.reset(static_cast<T*>(temp));
		allocType = AllocationType::CudaPinned;
		size = numElements;
	}

	void allocateStandard(const size_t numElements) {
		clear();
		ptr.reset(new T[numElements]);
		allocType = AllocationType::Standard;
		size = numElements;
	}

	void allocateAligned(const size_t numElements) {
		clear();
		ptr.reset(static_cast<T*>(_aligned_malloc(numElements * sizeof(T), 64)));
		allocType = AllocationType::Aligned;
		size = numElements;
	}

	void clear() {
		switch (allocType) {
			case AllocationType::Standard:
				ptr.reset();
				break;
			case AllocationType::Aligned:
				_aligned_free(ptr.get());
				break;
			case AllocationType::CudaPinned:
				cudaFreeHost(ptr.get());
				break;
			default:
				break;
		}
	}
};

template <typename CoordT, typename ValueT>
struct GridData {
	GridData() = default;

	// Simplified allocation functions
	void allocateCudaPinned(size_t numElements) {
		coordsBlock.allocateCudaPinned(numElements);
		valuesBlock.allocateCudaPinned(numElements);
		size = numElements;
	}

	void allocateStandard(size_t numElements) {
		coordsBlock.allocateStandard(numElements);
		valuesBlock.allocateStandard(numElements);
		size = numElements;
	}

	void allocateAligned(size_t numElements) {
		coordsBlock.allocateAligned(numElements);
		valuesBlock.allocateAligned(numElements);
		size = numElements;
	}

	void clear() {
		coordsBlock.clear();
		valuesBlock.clear();
		size = 0;
	}

	CoordT* pCoords() { return coordsBlock.ptr.get(); }
	const CoordT* pCoords() const { return coordsBlock.ptr.get(); }
	ValueT* pValues() { return valuesBlock.ptr.get(); }
	const ValueT* pValues() const { return valuesBlock.ptr.get(); }

	MemoryBlock<CoordT> coordsBlock{};
	MemoryBlock<ValueT> valuesBlock{};
	size_t size{0};
};


template <typename CoordType, typename ValueType>
using GenericGrid = GridData<CoordType, ValueType>;

template <typename T>
using OpenGrid = GenericGrid<openvdb::Coord, T>;
template <typename T>
using NanoGrid = GenericGrid<nanovdb::Coord, T>;

template <typename CoordType>
using FloatGrid = GenericGrid<CoordType, float>;
template <typename CoordType>
using VectorGrid = GenericGrid<CoordType, nanovdb::Vec3f>;

using OpenFloatGrid = OpenGrid<float>;
using OpenVectorGrid = OpenGrid<openvdb::Vec3f>;
using NanoFloatGrid = NanoGrid<float>;
using NanoVectorGrid = NanoGrid<nanovdb::Vec3f>;


}  // namespace HNS