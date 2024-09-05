//
// Created by zphrfx on 23/08/2024.
//

#pragma once

#include <UT/UT_StopWatch.h>

#include <utility>

class ScopedTimer {
   public:
	explicit ScopedTimer(std::string  name) : name_(std::move(name)) {
		watch_.start();

	}
	~ScopedTimer() {
		std::printf("%s Time: %f ms\n", name_.c_str(),  watch_.lap() * 1000.0);
	}

   private:
	const std::string name_;
	UT_StopWatch watch_;
};