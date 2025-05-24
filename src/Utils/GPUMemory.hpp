//
// Created by zphrfx on 09/04/2025.
//

#pragma once
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>


template <typename T>
struct DeviceMemory {
	T* ptr = nullptr;
	size_t count = 0;
	cudaStream_t stream_ = nullptr;  // Store stream for async free

	DeviceMemory() = default;  // Default constructor needed for map operations

	DeviceMemory(const size_t num_elements, cudaStream_t stream) : count(num_elements), stream_(stream) {
		if (count > 0) {
			cudaMallocAsync(&ptr, count * sizeof(T), stream_);
		}
	}

	// Disable copy constructor and assignment
	DeviceMemory(const DeviceMemory&) = delete;
	DeviceMemory& operator=(const DeviceMemory&) = delete;

	// Move constructor
	DeviceMemory(DeviceMemory&& other) noexcept : ptr(other.ptr), count(other.count), stream_(other.stream_) {
		other.ptr = nullptr;
		other.count = 0;
	}

	// Move assignment
	DeviceMemory& operator=(DeviceMemory&& other) noexcept {
		if (this != &other) {
			// Free existing resource if any
			if (ptr) {
				// Queue the free operation on the associated stream
				cudaFreeAsync(ptr, stream_);
			}
			// Transfer ownership
			ptr = other.ptr;
			count = other.count;
			stream_ = other.stream_;
			// Nullify the source object
			other.ptr = nullptr;
			other.count = 0;
		}
		return *this;
	}

	~DeviceMemory() {
		if (ptr) {
			cudaFreeAsync(ptr, stream_);
		}
	}

	T* get() const { return ptr; }
	size_t size() const { return count; }
	size_t bytes() const { return count * sizeof(T); }
};


// GpuMemoryManager encapsulates GPU memory fields by name.
// perform asynchronous copy operations between host and device, free memory, and create temporary fields.
class GpuMemoryManager {
   public:
	// Add a new field of type T with the given number of elements and CUDA stream.
	// Throws if a field with the given name already exists.
	~GpuMemoryManager() = default;
	template <typename T>
	void addField(const std::string& fieldName, size_t numElements, cudaStream_t stream) {
		if (fields.find(fieldName) != fields.end()) {
			throw std::runtime_error("Field already exists: " + fieldName);
		}
		fields[fieldName] = std::make_unique<FieldWrapper<T>>(numElements, stream);
	}

	// Retrieve the DeviceMemory of a registered field.
	// Throws if the field is not found or if the field type does not match T.
	template <typename T>
	DeviceMemory<T>& getField(const std::string& fieldName) {
		const auto it = fields.find(fieldName);
		if (it == fields.end()) {
			throw std::runtime_error("Field not found: " + fieldName);
		}
		FieldWrapper<T>* wrapper = dynamic_cast<FieldWrapper<T>*>(it->second.get());
		if (!wrapper) {
			throw std::runtime_error("Field type mismatch for field: " + fieldName);
		}
		return wrapper->mem;
	}


	// Copy data from host to device for a given field.
	// hostData must point to a buffer with at least as many bytes as the device memory.
	// Throws if the copy fails.
	template <typename T>
	void copyFromHost(const std::string& fieldName, const T* hostData, const cudaStream_t stream) {
		DeviceMemory<T>& mem = getField<T>(fieldName);
		const cudaError_t err = cudaMemcpyAsync(mem.get(), hostData, mem.bytes(), cudaMemcpyHostToDevice, stream);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMemcpyAsync (host to device) failed for field " + fieldName);
		}
	}

	// Copy data from device to host for a given field.
	// hostData must point to a buffer with at least as many bytes as the device memory.
	// Throws if the copy fails.
	template <typename T>
	void copyToHost(const std::string& fieldName, T* hostData, const cudaStream_t stream) {
		DeviceMemory<T>& mem = getField<T>(fieldName);
		const cudaError_t err = cudaMemcpyAsync(hostData, mem.get(), mem.bytes(), cudaMemcpyDeviceToHost, stream);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMemcpyAsync (device to host) failed for field " + fieldName);
		}
	}

	// Free a specific field, releasing its GPU memory.
	// After calling this function, the field is removed from the manager.
	void freeField(const std::string& fieldName) {
		const auto it = fields.find(fieldName);
		if (it == fields.end()) {
			throw std::runtime_error("Field not found: " + fieldName);
		}
		fields.erase(it);
	}

	// Optionally, free all fields (clear the map).
	void clear() { fields.clear(); }

	// Create a temporary field.
	// This is an alias for addField, but it returns a reference to the underlying DeviceMemory,
	// making it convenient to use immediately.
	template <typename T>
	DeviceMemory<T>& createTempField(const std::string& fieldName, const size_t numElements, const cudaStream_t stream) {
		addField<T>(fieldName, numElements, stream);
		return getField<T>(fieldName);
	}

   private:
	// Interface for our type-erased field wrapper.
	struct IFieldWrapper {
		virtual ~IFieldWrapper() = default;
	};

	// Template wrapper to store DeviceMemory of a specific type.
	template <typename T>
	struct FieldWrapper final : IFieldWrapper {
		FieldWrapper(size_t numElements, cudaStream_t stream) : mem(numElements, stream) {}
		DeviceMemory<T> mem;
	};

	// The map storing each field by name.
	std::unordered_map<std::string, std::unique_ptr<IFieldWrapper>> fields;
};
