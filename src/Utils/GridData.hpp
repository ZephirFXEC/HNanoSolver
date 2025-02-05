//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <nanovdb/NanoVDB.h>
#include <openvdb/Types.h>

#include "Memory.hpp"

namespace HNS {

template <typename CoordT, typename ValueT>
struct GridData {
	GridData() = default;

	bool allocateCudaPinned(size_t numElements) {
		if (!coordsBlock.allocateCudaPinned(numElements)) {
			std::cerr << "Failed to allocate coordinates block" << std::endl;
			return false;
		}
		if (!valuesBlock.allocateCudaPinned(numElements)) {
			std::cerr << "Failed to allocate values block" << std::endl;
			coordsBlock.clear();
			return false;
		}
		size = numElements;
		return true;
	}

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


/// @brief A grid data structure with indexed values blocks (e.g. density, velocity, etc.)
/// Data is stored in a contiguous memory block for each named value block. The coords are separate.
/// TODO(enzoc): implement GPU RLE encoding for the values blocks
struct GridIndexedData {
	GridIndexedData() = default;

	GridIndexedData(const GridIndexedData&) = delete;
	GridIndexedData& operator=(const GridIndexedData&) = delete;

	GridIndexedData(GridIndexedData&&) = default;
	GridIndexedData& operator=(GridIndexedData&&) = default;

	// Destructor clears memory
	~GridIndexedData() { clear(); }

	/// @brief Allocate memory for the coords block and fix 'size'
	/// @param numElements the number of elements
	/// @param mode the memory allocation mode (standard/aligned/cuda pinned)
	/// @return true on success
	bool allocateCoords(size_t numElements, const AllocationType mode) {
		clearIndexes();  // in case we re-allocate
		clearCoords();
		bool success = false;
		switch (mode) {
			case AllocationType::Standard:
				success = m_IndexBlock.allocateStandard(numElements);
				success = m_CoordBlock.allocateStandard(numElements);
				break;
			case AllocationType::Aligned:
				success = m_IndexBlock.allocateAligned(numElements);
				success = m_CoordBlock.allocateAligned(numElements);
				break;
			case AllocationType::CudaPinned:
				success = m_IndexBlock.allocateCudaPinned(numElements);
				success = m_CoordBlock.allocateCudaPinned(numElements);
				break;
		}
		if (!success) {
			std::cerr << "Failed to allocate coordinate block!\n";
			return false;
		}
		m_size = numElements;
		return true;
	}


	/// @brief Add a block of type T with 'numElements' (usually matches coords size)
	/// @param name  e.g. "density", "velocity", etc.
	/// @param mode  allocation mode
	/// @param numElements  how many elements in this block
	/// @return true if success, false if allocation fails or name is duplicate
	template <typename T>
	bool addValueBlock(const std::string& name, AllocationType mode, size_t numElements) {
		if (findBlockIndex(name) >= 0) {
			std::cerr << "Block with name '" << name << "' already exists!\n";
			return false;
		}

		// Create a typed block
		auto typedBlock = std::make_unique<TypedValueBlock<T>>();
		if (!typedBlock->allocate(numElements, mode)) {
			std::cerr << "Failed to allocate typed block for '" << name << "'\n";
			return false;
		}

		// Store in our vector
		m_valueBlocks.emplace_back(std::move(typedBlock), name);
		return true;
	}

	/// @brief Get a typed block by name. Returns null if not found or type mismatch.
	template <typename T>
	TypedValueBlock<T>* getValueBlock(const std::string& name) {
		const int idx = findBlockIndex(name);
		if (idx < 0) {
			return nullptr;
		}
		// Attempt dynamic_cast
		auto* ptr = dynamic_cast<TypedValueBlock<T>*>(m_valueBlocks[idx].first.get());
		return ptr;  // could be null if the block is not actually T
	}

	/// @return pointer to the Index array (null if unallocated)
	[[nodiscard]] uint64_t* pIndexes() { return m_IndexBlock.ptr.get(); }
	[[nodiscard]] const uint64_t* pIndexes() const { return m_IndexBlock.ptr.get(); }

	/// @return pointer to the Coord array (null if unallocated)
	[[nodiscard]] openvdb::Coord* pCoords() { return m_CoordBlock.ptr.get(); }
	[[nodiscard]] const openvdb::Coord* pCoords() const { return m_CoordBlock.ptr.get(); }

	/// @brief Convenience function: returns the data pointer for block <T> with given name
	///        or nullptr if not found / mismatch.
	template <typename T>
	T* pValues(const std::string& name) {
		auto* block = getValueBlock<T>(name);
		return block ? block->data() : nullptr;
	}

	/// @brief Convenience function: returns the data pointer for block <T> with given name
	///        or nullptr if not found / mismatch.
	template <typename T>
	const T* pValues(const std::string& name) const {
		auto* block = getValueBlock<T>(name);
		return block ? block->data() : nullptr;
	}

	/// @return how many elements are allocated
	[[nodiscard]] size_t size() const { return m_size; }

	/// @return the number of distinct named value blocks
	[[nodiscard]] size_t numValueBlocks() const { return m_valueBlocks.size(); }

	/// @return the name of the i-th value block
	[[nodiscard]] const std::string& blockName(const size_t i) const { return m_valueBlocks[i].second; }


	/// @brief Deallocate everything
	void clear() {
		clearIndexes();
		clearValues();
		clearCoords();
		m_size = 0;
	}

	/// @brief Deallocate only the coords block
	void clearIndexes() { m_IndexBlock.clear(); }

	/// @brief Deallocate only the values blocks
	void clearValues() {
		for (auto& [fst, snd] : m_valueBlocks) {
			fst->clear();
		}
		m_valueBlocks.clear();
	}

	void clearCoords() { m_CoordBlock.clear(); }

   private:
	/// @brief Find index of block by name, or -1 if none
	[[nodiscard]] int findBlockIndex(const std::string& name) const {
		for (size_t i = 0; i < m_valueBlocks.size(); ++i) {
			if (m_valueBlocks[i].second == name) {
				return static_cast<int>(i);
			}
		}
		return -1;
	}

	// The Index block (one array)
	MemoryBlock<uint64_t> m_IndexBlock{};
	MemoryBlock<openvdb::Coord> m_CoordBlock{};
	size_t m_size{0};

	// A list of (valueBlock, name). Each valueBlock is an IValueBlock (type-erased).
	// The 'std::unique_ptr<IValueBlock>' can point to a TypedValueBlock<T> of any T.
	std::vector<std::pair<std::unique_ptr<IValueBlock>, std::string>> m_valueBlocks{};
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