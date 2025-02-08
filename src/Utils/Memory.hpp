//
// Created by zphrfx on 24/10/2024.
//

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <typeinfo>

enum class AllocationType : uint8_t { Standard, Aligned, CudaPinned };

template <typename T>
struct MemoryBlock {
	struct Deleter {
		AllocationType allocType = AllocationType::Standard;
		__forceinline void operator()(T* ptr) const noexcept {
			if (!ptr) return;
			switch (allocType) {
				case AllocationType::Standard:
					delete[] ptr;
					break;
				case AllocationType::Aligned:
					_aligned_free(ptr);
					break;
				case AllocationType::CudaPinned:
					cudaFreeHost(ptr);
					break;
				default:
					break;
			}
		}
	};

	std::unique_ptr<T[], Deleter> ptr{nullptr};
	size_t size = 0;

	MemoryBlock() = default;
	MemoryBlock(MemoryBlock&&) noexcept = default;
	MemoryBlock& operator=(MemoryBlock&&) noexcept = default;
	MemoryBlock(const MemoryBlock&) = delete;
	MemoryBlock& operator=(const MemoryBlock&) = delete;
	~MemoryBlock() = default;

	bool allocateCudaPinned(const size_t numElements) noexcept {
		clear();
		void* temp = nullptr;
		const cudaError_t err = cudaMallocHost(&temp, numElements * sizeof(T));
		if (err != cudaSuccess) {
			std::cerr << "Error allocating pinned memory: " << cudaGetErrorString(err) << std::endl;
			return false;
		}
		ptr.reset(static_cast<T*>(temp));
		ptr.get_deleter().allocType = AllocationType::CudaPinned;
		size = numElements;
		return true;
	}

	bool allocateStandard(const size_t numElements) noexcept {
		clear();
		T* temp = new T[numElements];
		if (!temp) {
			std::cerr << "Error allocating standard memory" << std::endl;
			return false;
		}
		ptr.reset(temp);
		ptr.get_deleter().allocType = AllocationType::Standard;
		size = numElements;
		return true;
	}

	bool allocateAligned(const size_t numElements) noexcept {
		clear();
		T* temp = static_cast<T*>(_aligned_malloc(numElements * sizeof(T), 64));
		if (!temp) {
			std::cerr << "Error allocating aligned memory" << std::endl;
			return false;
		}
		ptr.reset(temp);
		ptr.get_deleter().allocType = AllocationType::Aligned;
		size = numElements;
		return true;
	}

	void clear() noexcept {
		if (ptr) {
			ptr.reset();
			size = 0;
		}
	}
};


/// @brief Abstract base class for any typed memory block
struct IValueBlock {
	virtual ~IValueBlock() = default;
	virtual void clear() noexcept = 0;
	[[nodiscard]] virtual size_t size() const = 0;
	[[nodiscard]] virtual const std::type_info& typeInfo() const = 0;
};


template <typename T>
struct TypedValueBlock final : IValueBlock {
	MemoryBlock<T> block;

	TypedValueBlock() = default;
	~TypedValueBlock() override { clear(); }

	T& operator[](size_t i) { return block.ptr[i]; }
	const T& operator[](size_t i) const { return block.ptr[i]; }

	/// @brief Allocate the block using the requested allocation mode.
	bool allocate(size_t numElements, AllocationType mode) noexcept {
		clear();
		bool success = false;
		switch (mode) {
			case AllocationType::Standard:
				success = block.allocateStandard(numElements);
				break;
			case AllocationType::Aligned:
				success = block.allocateAligned(numElements);
				break;
			case AllocationType::CudaPinned:
				success = block.allocateCudaPinned(numElements);
				break;
			default:
				break;
		}
		return success;
	}

	void clear() noexcept override { block.clear(); }
	[[nodiscard]] size_t size() const override { return block.size; }
	[[nodiscard]] const std::type_info& typeInfo() const override { return typeid(T); }
	[[nodiscard]] T* data() { return block.ptr.get(); }
	[[nodiscard]] const T* data() const { return block.ptr.get(); }
};