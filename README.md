# HNanoSolver
> This is a project used to learn Fluid Dynamics and GPU acceleration through Cuda.

# DFD : 
![image](https://github.com/user-attachments/assets/2a453b9c-edbc-4487-b3fd-368d56098b4d)

Converting back and forth from OpenVDB to NanoVDB is expensive, around 150ms each with 6M active voxel with a 1080ti. 
To avoid those conversions, the idea is to load only the sourcing at every frame which are usually smaller VDBs, and add those sourcing VDBs to the existing data in the GPU.
We can then decide which grid we want to build in Houdini at every frame (usually the density). this grid has to be converted from NanoVDB to OpenVDB. 

For a simple simulation with a Density grid and Velocity grid with 6M active voxels for each grids, this avoids having to load at every frame the newly build Density / Velocity ~300ms and having to build those grid back in Houdini, ~300ms.

- Dummy implementation : ~600ms/frame. 
- New Idea : 150ms/frame.  



## How to Compile : 
> To comply with Houdini 20.5.278, I use Visual Studio 2022 with v142 compiler version.
> Make sure you have at least Cuda 12.2 install.
- Clone the repo : `git pull https://github.com/ZephirFXEC/HNanoSolver.git`
- Create a build directory : `cd HNanoSolver && mkdir build && cd build`
- Build using cmake : `cmake .. -G "Visual Studio 17 2022" -T "v142"`

# Reference
## Papers 
- Bridson, R., Muller-Fischer, M. (2007). FLUID SIMULATION SIGGRAPH 2007 Course notes. SIGGRAPH. [this](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)
  
