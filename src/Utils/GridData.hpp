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

	template <typename MemoryT>
	struct MemoryBlock {
		MemoryT* ptr = nullptr;
		AllocationType allocType = AllocationType::None;
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
		if (err != cudaSuccess) {
			throw std::runtime_error("Failed to allocate pinned memory for coords");
		}

		err = cudaMallocHost(&valuesBlock.ptr, numElements * sizeof(ValueT));
		if (err != cudaSuccess) {
			coordsBlock.free();
			throw std::runtime_error("Failed to allocate pinned memory for values");
		}

		valuesBlock.allocType = AllocationType::CudaPinned;
		coordsBlock.allocType = AllocationType::CudaPinned;

		size = numElements;
	}

	void allocateAligned(const size_t numElements, size_t alignment) {
		clear();
		coordsBlock.ptr = static_cast<CoordT*>(_aligned_malloc(numElements * sizeof(CoordT), alignment));
		if (!coordsBlock.ptr) throw std::runtime_error("Failed to allocate aligned memory for coords");

		valuesBlock.ptr = static_cast<ValueT*>(_aligned_malloc(numElements * sizeof(ValueT), alignment));
		if (!valuesBlock.ptr) {
			coordsBlock.free();
			throw std::runtime_error("Failed to allocate aligned memory for values");
		}

		valuesBlock.allocType = AllocationType::AlignedMalloc;
		valuesBlock.alignment = alignment;
		coordsBlock.allocType = AllocationType::AlignedMalloc;
		coordsBlock.alignment = alignment;
		size = numElements;
	}

	void allocateStandard(const size_t numElements) {
		clear();

		coordsBlock.ptr = static_cast<CoordT*>(malloc(numElements * sizeof(CoordT)));
		if (!coordsBlock.ptr) throw std::runtime_error("Failed to allocate memory for coords");

		valuesBlock.ptr = static_cast<ValueT*>(malloc(numElements * sizeof(ValueT)));
		if (!valuesBlock.ptr) {
			coordsBlock.free();
			throw std::runtime_error("Failed to allocate memory for values");
		}

		valuesBlock.allocType = AllocationType::StandardMalloc;
		coordsBlock.allocType = AllocationType::StandardMalloc;

		size = numElements;
	}

	CoordT* pCoords() { return coordsBlock.ptr; }
	const CoordT* pCoords() const { return coordsBlock.ptr; }

	ValueT* pValues() { return valuesBlock.ptr; }
	const ValueT* pValues() const { return valuesBlock.ptr; }

	// Clear all memory
	void clear() {
		coordsBlock.free();
		valuesBlock.free();
		size = 0;
	}

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