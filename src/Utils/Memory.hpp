#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <typeinfo>

#ifdef __linux__
#include <malloc.h>
#define _aligned_malloc(size, alignment) memalign(alignment, size)
#define _aligned_free free
#elif _WIN32
#include <malloc.h>
#define _aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define _aligned_free _aligned_free
#else

#endif

enum class AllocationType { Standard, Aligned, CudaPinned };

template <typename T>
struct MemoryBlock {
	MemoryBlock() = default;

	~MemoryBlock() { clear(); }

	bool allocateCudaPinned(const size_t numElements) {
		clear();
		void* temp = nullptr;
		if (const cudaError_t err = cudaMallocHost(&temp, numElements * sizeof(T)); err != cudaSuccess) {
			std::cerr << "Error allocating pinned memory: " << cudaGetErrorString(err) << std::endl;
			return false;
		}
		ptr.reset(static_cast<T*>(temp));
		ptr.get_deleter().allocType = AllocationType::CudaPinned;
		size = numElements;
		return true;
	}

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

	void clear() {
		ptr.reset();
		size = 0;
	}

	struct Deleter {
		AllocationType allocType = AllocationType::Standard;

		void operator()(T* ptr) {
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

	std::unique_ptr<T[], Deleter> ptr{nullptr, Deleter{AllocationType::Standard}};
	size_t size = 0;
};


/// @brief Abstract base class for any typed memory block
struct IValueBlock {
	virtual ~IValueBlock() = default;

	/// Deallocate memory if allocated
	virtual void clear() = 0;

	/// Number of elements in this block
	[[nodiscard]] virtual size_t size() const = 0;

	/// Return the std::type_info of this block's element type
	[[nodiscard]] virtual const std::type_info& typeInfo() const = 0;
};


template <typename T>
struct TypedValueBlock final : IValueBlock {
	MemoryBlock<T> block;
	size_t m_size = 0;

	TypedValueBlock() = default;
	~TypedValueBlock() override { TypedValueBlock::clear(); }

	T& operator[](size_t i) { return block.ptr[i]; }

	const T& operator[](size_t i) const { return block.ptr[i]; }

	/// @brief Allocate the block with the chosen memory mode
	bool allocate(size_t numElements, const AllocationType mode) {
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
		}
		if (success) {
			m_size = numElements;
		}
		return success;
	}

	/// @brief IValueBlock interface
	void clear() override {
		block.clear();
		m_size = 0;
	}

	/// @brief IValueBlock interface
	[[nodiscard]] size_t size() const override { return m_size; }

	/// @brief IValueBlock interface
	[[nodiscard]] const std::type_info& typeInfo() const override { return typeid(T); }

	/// @brief Return the raw pointer
	T* data() { return block.ptr.get(); }
	const T* data() const { return block.ptr.get(); }
};