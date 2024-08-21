# HNanoSolver
> This is a project used to learn Fluid Dynamics and GPU acceleration through Cuda.

## How to Compile : 
> To comply with Houdini 20.5.278, I use Visual Studio 2022 with v142 compiler version.
> Make sure you have at least Cuda 12.2 install.
- Clone the repo : `git pull https://github.com/ZephirFXEC/HNanoSolver.git`
- Create a build directory : `cd HNanoSolver && mkdir build && cd build`
- Build using cmake : `cmake .. -G "Visual Studio 17 2022" -T "v142"`
> Note that I modified my Houdini CMake file, and defaulted it to link against the `openvdb_sesi.lib` avaible in Houdini.
