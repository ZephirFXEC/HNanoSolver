#ifndef NANOVDB_CUDA_PYSIGNEDFLOODFILL_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYSIGNEDFLOODFILL_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BuildT> void defineSignedFloodFill(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
