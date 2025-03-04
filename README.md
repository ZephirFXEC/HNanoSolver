
# [HNanoSolver](https://youtu.be/W5Qsye3BMng)
> This is a project used to learn Fluid Dynamics and GPU acceleration through Cuda.
- **HNanoSolver**, All in one node computing a quantity advection by a divergence free velocity field on the GPU.
- **HNanoProjectNonDivergent**, Compute a Divergence free velocity field on the GPU.
- **HNanoAdvect**, Advect any float fields on the GPU.
- **HNanoAdvectVelocity**, Advect any vector field on the GPU.
- **HNanoViewer**, Standalone app to visualize VDBs and kernels without having to relaunch Houdini.


# Warning ! 
> Very Early Development 
This repo is undergoing rewrite very often and might not compile from time to time.
Feel free to check the [Open Issues](https://github.com/ZephirFXEC/HNanoSolver/issues) for more details.


# How to Build : 
## Windows : 
> Make sure HFS is set in your PATH.
- `git clone https://github.com/ZephirFXEC/HNanoSolver`
- `cd HNanoSolver && mkdir build && cd build`
- `cmake -G "Visual Studio 17 2022" -T v142`
It will generate Visual Studio solutions files that you can use to Compile & Run.

## Linux : 
> Tested on Ubuntu 24 using GCC 13.3 and CUDA 12.8
- `git clone https://github.com/ZephirFXEC/HNanoSolver`
- `cd HNanoSolver` 
- `mkdir build && cd build`
- `cmake -DCMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_FLAGS:STRING="-O2 -g -DNDEBUG -fPIC" ..`
> if any errors here, make sure to have $HFS set, and the right NVCC compiler
- `make`
> if any errors related to AVX / Cuda compilation it's probably due to an incompatibility between GCC and NVCC



# Reference
## Papers 
- Bridson, R., Muller-Fischer, M. (2007). FLUID SIMULATION SIGGRAPH 2007 Course notes. SIGGRAPH. [this](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)
- Gabriel D. Weymouth, Data-driven Multi-Grid solver for accelerated pressure projection, [this](https://www.sciencedirect.com/science/article/pii/S0045793022002213)
- Williams and Others, fVDB: A Deep-Learning Framework for Sparse, Large-Scale, and High-Performance Spatial Intelligence, [this](https://research.nvidia.com/labs/prl/williams2024fVDB/fVDB.pdf)


# Pictures 

# HNanoViewer : 
![image](https://github.com/user-attachments/assets/82402e68-e462-4932-83d9-3b63219403a6)


# Still Frame : 
![image](https://github.com/user-attachments/assets/12de0c85-87df-4b12-ab81-4973c024d9e0)
