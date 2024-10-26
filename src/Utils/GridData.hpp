//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <nanovdb/NanoVDB.h>
#include <openvdb/openvdb.h>
namespace HNS {
static constexpr size_t ALIGNMENT = 32;
template <typename CoordT, typename ValueT>
struct GridData {
	enum class AllocationType {
		None,
		CudaPinned,
		AlignedMalloc,
		StandardMalloc,
	};

	struct MemoryBlock {
		void* ptr = nullptr;
		AllocationType allocType = AllocationType::None;
		size_t size = 0;
		size_t alignment = 0;

		void free() {
			if (!ptr) return;
			switch (allocType) {
				case AllocationType::CudaPinned:
					cudaFreeHost(ptr);
					break;
				case AllocationType::AlignedMalloc:
					_aligned_free(ptr);
					break;
				case AllocationType::StandardMalloc:
					std::free(ptr);
					break;
				case AllocationType::None:
					break;
			}
			ptr = nullptr;
			size = 0;
			allocType = AllocationType::None;
		}
	};

	// Destructor
	~GridData() { clear(); }

	// Move operations
	GridData(GridData&& other) noexcept
	    : coordsBlock(other.coordsBlock), valuesBlock(other.valuesBlock), size(other.size) {
		other.coordsBlock = MemoryBlock();
		other.valuesBlock = MemoryBlock();
		other.size = 0;
	}

	GridData& operator=(GridData&& other) noexcept {
		if (this != &other) {
			clear();
			coordsBlock = other.coordsBlock;
			valuesBlock = other.valuesBlock;
			size = other.size;
			other.coordsBlock = MemoryBlock();
			other.valuesBlock = MemoryBlock();
			other.size = 0;
		}
		return *this;
	}

	// Delete copy operations
	GridData(const GridData&) = delete;
	GridData& operator=(const GridData&) = delete;

	// Default constructor
	GridData() = default;

	void allocateCudaPinned(const size_t numElements) {
		clear();  // Free existing memory first

		cudaError_t err = cudaMallocHost(&coordsBlock.ptr, numElements * sizeof(CoordT));
		if (err != cudaSuccess) throw std::runtime_error("Failed to allocate pinned memory for coords");
		coordsBlock.allocType = AllocationType::CudaPinned;
		coordsBlock.size = numElements * sizeof(CoordT);

		err = cudaMallocHost(&valuesBlock.ptr, numElements * sizeof(ValueT));
		if (err != cudaSuccess) {
			coordsBlock.free();
			throw std::runtime_error("Failed to allocate pinned memory for values");
		}
		valuesBlock.allocType = AllocationType::CudaPinned;
		valuesBlock.size = numElements * sizeof(ValueT);

		size = numElements;
	}

	void allocateAligned(const size_t numElements, size_t alignment) {
		clear();
		coordsBlock.ptr = _aligned_malloc(numElements * sizeof(CoordT), alignment);

		if (!coordsBlock.ptr) throw std::runtime_error("Failed to allocate aligned memory for coords");
		coordsBlock.allocType = AllocationType::AlignedMalloc;
		coordsBlock.size = numElements * sizeof(CoordT);
		coordsBlock.alignment = alignment;
		valuesBlock.ptr = _aligned_malloc(numElements * sizeof(ValueT), alignment);
		if (!valuesBlock.ptr) {
			coordsBlock.free();
			throw std::runtime_error("Failed to allocate aligned memory for values");
		}
		valuesBlock.allocType = AllocationType::AlignedMalloc;
		valuesBlock.size = numElements * sizeof(ValueT);
		valuesBlock.alignment = alignment;

		size = numElements;
	}

	void allocateStandard(const size_t numElements) {
		clear();

		coordsBlock.ptr = malloc(numElements * sizeof(CoordT));
		if (!coordsBlock.ptr) throw std::runtime_error("Failed to allocate memory for coords");
		coordsBlock.allocType = AllocationType::StandardMalloc;
		coordsBlock.size = numElements * sizeof(CoordT);

		valuesBlock.ptr = malloc(numElements * sizeof(ValueT));
		if (!valuesBlock.ptr) {
			coordsBlock.free();
			throw std::runtime_error("Failed to allocate memory for values");
		}
		valuesBlock.allocType = AllocationType::StandardMalloc;
		valuesBlock.size = numElements * sizeof(ValueT);

		size = numElements;
	}

	CoordT* pCoords() { return static_cast<CoordT*>(coordsBlock.ptr); }
	const CoordT* pCoords() const { return static_cast<const CoordT*>(coordsBlock.ptr); }

	ValueT* pValues() { return static_cast<ValueT*>(valuesBlock.ptr); }
	const ValueT* pValues() const { return static_cast<const ValueT*>(valuesBlock.ptr); }

	// Clear all memory
	void clear() {
		coordsBlock.free();
		valuesBlock.free();
		size = 0;
	}

	MemoryBlock coordsBlock{};
	MemoryBlock valuesBlock{};
	size_t size{0};
};

// Base templates for OpenVDB and NanoVDB grids
template <typename CoordType, typename ValueType>
using GenericGrid = GridData<CoordType, ValueType>;

template <typename T>
using OpenGrid = GenericGrid<openvdb::Coord, T>;

template <typename T>
using NanoGrid = GenericGrid<nanovdb::Coord, T>;

// Specific grid types with float and vector values
template <typename CoordType>
using FloatGrid = GenericGrid<CoordType, float>;

template <typename CoordType>
using VectorGrid = GenericGrid<CoordType, nanovdb::Vec3f>;

// Typedefs for common OpenVDB and NanoVDB grids
using OpenFloatGrid = OpenGrid<float>;
using OpenVectorGrid = OpenGrid<openvdb::Vec3f>;

using NanoFloatGrid = NanoGrid<float>;
using NanoVectorGrid = NanoGrid<nanovdb::Vec3f>;


}  // namespace HNS