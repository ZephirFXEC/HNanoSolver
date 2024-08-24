//
// Created by zphrfx on 23/08/2024.
//

#pragma once

#include <chrono>

class ScopedTimer {
public:
	explicit ScopedTimer(const char* name) : name_(name), start_(std::chrono::system_clock::now()) {}
	~ScopedTimer() {
		const auto end = std::chrono::system_clock::now();
		std::printf("%s Time: %f ms\n", name_, std::chrono::duration<double>(end - start_).count()*100.0);
	}
private:
	const char* name_;
	const std::chrono::time_point<std::chrono::system_clock> start_;
};