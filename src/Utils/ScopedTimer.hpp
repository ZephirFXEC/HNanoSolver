//
// Created by zphrfx on 23/08/2024.
//

#pragma once

#include <UT/UT_StopWatch.h>

class ScopedTimer {
   public:
	explicit ScopedTimer(const char* name) : name_(name) {
		watch_.start();

	}
	~ScopedTimer() {
		std::printf("%s Time: %f ms\n", name_,  watch_.lap() * 1000.0);
	}

   private:
	const char* name_;
	UT_StopWatch watch_;
};