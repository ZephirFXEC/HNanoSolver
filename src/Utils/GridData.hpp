//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <nanovdb/NanoVDB.h>
#include <openvdb/Types.h>

#include <iostream>



namespace HNS {

enum class AllocationType { Standard, Aligned };

template <typename T>
struct MemoryBlock {
	struct Deleter {
		AllocationType allocType = AllocationType::Standard;

		void operator()(T* ptr) {
			switch (allocType) {
				case AllocationType::Standard:
					delete[] ptr;
					break;
				case AllocationType::Aligned:
					free(ptr);
					break;
				default:
					break;
			}
		}
	};

	std::unique_ptr<T[], Deleter> ptr{nullptr, Deleter{AllocationType::Standard}};
	size_t size = 0;

	MemoryBlock() = default;

	bool allocateStandard(const size_t numElements) {
		clear();
		try {
			ptr.reset(new T[numElements]);
			ptr.get_deleter().allocType = AllocationType::Standard;
			size = numElements;
			return true;
		} catch (const std::bad_alloc&) {
			std::cerr << "Error allocating standard memory" << std::endl;
			return false;
		}
	}

	bool allocateAligned(const size_t numElements) {
		clear();
		T* temp = static_cast<T*>(aligned_alloc(numElements * sizeof(T), 64));
		if (!temp) {
			std::cerr << "Error allocating aligned memory" << std::endl;
			return false;
		}
		ptr.reset(temp);
		ptr.get_deleter().allocType = AllocationType::Aligned;
		size = numElements;
		return true;
	}

	void clear() {
		ptr.reset();
		size = 0;
	}
};

template <typename CoordT, typename ValueT>
struct GridData {
	GridData() = default;

	bool allocateStandard(size_t numElements) {
		if (!coordsBlock.allocateStandard(numElements)) {
			std::cerr << "Failed to allocate coordinates block" << std::endl;
			return false;
		}
		if (!valuesBlock.allocateStandard(numElements)) {
			std::cerr << "Failed to allocate values block" << std::endl;
			coordsBlock.clear();
			return false;
		}
		size = numElements;
		return true;
	}

	bool allocateAligned(size_t numElements) {
		if (!coordsBlock.allocateAligned(numElements)) {
			std::cerr << "Failed to allocate coordinates block" << std::endl;
			return false;
		}
		if (!valuesBlock.allocateAligned(numElements)) {
			std::cerr << "Failed to allocate values block" << std::endl;
			coordsBlock.clear();
			return false;
		}
		size = numElements;
		return true;
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