//
// Created by zphrfx on 15/09/2024.
//
#pragma once

#include <typeindex>
#include <nanovdb/NanoVDB.h>
#include "Memory.hpp"

namespace HNS {

/// @brief A grid data structure with indexed values blocks (e.g. density, velocity, etc.)
/// Data is stored in a contiguous memory block for each named value block. The coords are separate.
/// TODO(enzoc): implement GPU RLE encoding for the values blocks
struct GridIndexedData {
	GridIndexedData() = default;

	GridIndexedData(const GridIndexedData&) = delete;
	GridIndexedData& operator=(const GridIndexedData&) = delete;

	GridIndexedData(GridIndexedData&&) noexcept = default;
	GridIndexedData& operator=(GridIndexedData&&) = default;

	// Destructor clears memory
	~GridIndexedData() { clear(); }

	/// @brief Allocate memory for the coords block and fix 'size'
	/// @param numElements the number of elements
	/// @param mode the memory allocation mode (standard/aligned/cuda pinned)
	/// @return true on success
	bool allocateCoords(size_t numElements) {
		clearIndexes();
		clearCoords();

		auto allocHelper = [&](auto&& indexAlloc, auto&& coordAlloc) -> bool {
			if (!(m_IndexBlock.*indexAlloc)(numElements)) return false;
			if (!(m_CoordBlock.*coordAlloc)(numElements)) {
				m_IndexBlock.clear();
				return false;
			}
			return true;
		};

		bool success;
		switch (m_allocType) {
			case AllocationType::Standard:
				success = allocHelper(&MemoryBlock<uint64_t>::allocateStandard, &MemoryBlock<nanovdb::Coord>::allocateStandard);
				break;
			case AllocationType::Aligned:
				success = allocHelper(&MemoryBlock<uint64_t>::allocateAligned, &MemoryBlock<nanovdb::Coord>::allocateAligned);
				break;
			case AllocationType::CudaPinned:
				success = allocHelper(&MemoryBlock<uint64_t>::allocateCudaPinned, &MemoryBlock<nanovdb::Coord>::allocateCudaPinned);
				break;
			default:
				success = false;
		}

		if (success)
			m_size = numElements;
		else
			std::cerr << "Failed to allocate coordinate block!\n";
		return success;
	}


	/// @brief Add a block of type T with 'numElements' (usually matches coords size)
	/// @param name  e.g. "density", "velocity", etc.
	/// @param mode  allocation mode
	/// @param numElements  how many elements in this block
	/// @return true if success, false if allocation fails or name is duplicate
	template <typename T>
	bool addValueBlock(const std::string& name, size_t numElements) {
		if (m_blockNameMap.find(name) != m_blockNameMap.end()) {
			std::cerr << "Block with name '" << name << "' already exists!\n";
			return false;
		}

		auto typedBlock = std::make_unique<TypedValueBlock<T>>();
		if (!typedBlock->allocate(numElements, m_allocType)) {
			std::cerr << "Failed to allocate typed block for '" << name << "'\n";
			return false;
		}

		m_valueBlocks.emplace_back(ValueBlockEntry{std::move(typedBlock), name, std::type_index(typeid(T))});
		m_blockNameMap[name] = m_valueBlocks.size() - 1;
		return true;
	}

	/// @brief Get a typed block by name. Returns null if not found or type mismatch.
	template <typename T>
	TypedValueBlock<T>* getValueBlock(const std::string& name) {
		const auto it = m_blockNameMap.find(name);
		if (it == m_blockNameMap.end()) return nullptr;

		const auto& entry = m_valueBlocks[it->second];
		if (entry.type != std::type_index(typeid(T))) return nullptr;

		return static_cast<TypedValueBlock<T>*>(entry.block.get());
	}

	/// @return pointer to the Index array (null if unallocated)
	[[nodiscard]] __forceinline uint64_t* pIndexes() { return m_IndexBlock.ptr.get(); }
	[[nodiscard]] __forceinline const uint64_t* pIndexes() const { return m_IndexBlock.ptr.get(); }
	[[nodiscard]] __forceinline nanovdb::Coord* pCoords() { return m_CoordBlock.ptr.get(); }
	[[nodiscard]] __forceinline const nanovdb::Coord* pCoords() const { return m_CoordBlock.ptr.get(); }

	/// Convenience: return the data pointer for a value block (or nullptr on error)
	template <typename T>
	__forceinline T* pValues(const std::string& name) {
		auto* block = getValueBlock<T>(name);
		return block ? block->data() : nullptr;
	}
	template <typename T>
	__forceinline const T* pValues(const std::string& name) const {
		auto* block = getValueBlock<T>(name);
		return block ? block->data() : nullptr;
	}

	[[nodiscard]] __forceinline size_t size() const { return m_size; }
	[[nodiscard]] __forceinline size_t numValueBlocks() const { return m_valueBlocks.size(); }

	__forceinline void clear() noexcept {
		clearIndexes();
		clearValues();
		clearCoords();
		m_size = 0;
	}
	__forceinline void clearIndexes() noexcept { m_IndexBlock.clear(); }
	__forceinline void clearCoords() noexcept { m_CoordBlock.clear(); }
	__forceinline void clearValues() noexcept {
		m_valueBlocks.clear();
		m_blockNameMap.clear();
	}

	void setAllocationType(const AllocationType type) { m_allocType = type; }

   private:
	/// @brief Find index of block by name, or -1 if none
	[[nodiscard]] int findBlockIndex(const std::string& name) const {
		const auto it = m_blockNameMap.find(name);
		return it != m_blockNameMap.end() ? static_cast<int>(it->second) : -1;
	}

	// The Index block (one array)
	MemoryBlock<uint64_t> m_IndexBlock{};
	MemoryBlock<nanovdb::Coord> m_CoordBlock{};
	size_t m_size{0};

	struct ValueBlockEntry {
		std::unique_ptr<IValueBlock> block;
		std::string name;
		std::type_index type;
	};

	std::vector<ValueBlockEntry> m_valueBlocks{};
	std::unordered_map<std::string, size_t> m_blockNameMap{};
	AllocationType m_allocType{AllocationType::Standard};
};
}  // namespace HNS