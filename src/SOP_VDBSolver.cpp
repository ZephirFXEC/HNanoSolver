#include "SOP_VDBSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/VolumeAdvect.h>

using namespace VdbSolver;


extern "C" void launch_kernels(nanovdb::NanoGrid<float>*, const nanovdb::NanoGrid<nanovdb::Vec3f>*, int, float,
                               cudaStream_t stream);


void newSopOperator(OP_OperatorTable* table) {
	if (table == nullptr) return;

	houdini_utils::ParmList parms;

	parms.add(houdini_utils::ParmFactory(PRM_TOGGLE, "debug", "Print Debug Information")
	              .setDefault(PRMoneDefaults)
	              .setTooltip("Print debug information to the console")
	              .setDocumentation("Print debug information to the console"));

	parms.add(houdini_utils::ParmFactory(PRM_TOGGLE, "GPU", "Runs on the GPU")
	              .setDefault(PRMoneDefaults)
	              .setTooltip("Runs advection on the GPU using NanoVDB"));

	// Level set grid
	parms.add(houdini_utils::ParmFactory(PRM_STRING, "group", "Group")
	              .setChoiceList(&houdini_utils::PrimGroupMenuInput1)
	              .setTooltip("VDB grid(s) to advect.")
	              .setDocumentation("A subset of VDBs in the first input to move using the velocity field"
	                                " (see [specifying volumes|/model/volumes#group])"));

	// Velocity grid
	parms.add(houdini_utils::ParmFactory(PRM_STRING, "velgroup", "Velocity")
	              .setChoiceList(&houdini_utils::PrimGroupMenuInput2)
	              .setTooltip("Velocity grid")
	              .setDocumentation("The name of a VDB primitive in the second input to use as"
	                                " the velocity field (see [specifying volumes|/model/volumes#group])\n\n"
	                                "This must be a vector-valued VDB primitive."
	                                " You can use the [Vector Merge node|Node:sop/DW_OpenVDBVectorMerge]"
	                                " to turn a `vel.[xyz]` triple into a single primitive."));

	// Advect: timestep
	parms.add(houdini_utils::ParmFactory(PRM_FLT, "timestep", "Timestep")
	              .setDefault(1, "1.0/$FPS")
	              .setDocumentation("Number of seconds of movement to apply to the input points\n\n"
	                                "The default is `1/$FPS` (one frame's worth of time)."
	                                " You can use negative values to move the points backwards through"
	                                " the velocity field."));


	// Register this operator.
	OpenVDBOpFactory("HNanoAdvect", SOP_VdbSolver::myConstructor, parms, *table)
	    .setNativeName("hnanoadvect")
	    .addInput("VDBs to Advect")
	    .addInput("Velocity VDB")
	    .setVerb(SOP_NodeVerb::COOK_INPLACE, [] { return new SOP_VdbSolver::Cache; });
}

const char* SOP_VdbSolver::inputLabel(const unsigned idx) const {
	switch (idx) {
		case 0:
			return "Input Grids";

		case 1:
			return "Velocity Grids";
		default:
			return "default";
	}
}

OP_Node* SOP_VdbSolver::myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
	return new SOP_VdbSolver(net, name, op);
}

SOP_VdbSolver::SOP_VdbSolver(OP_Network* net, const char* name, OP_Operator* op) : SOP_NodeVDB(net, name, op) {}

SOP_VdbSolver::~SOP_VdbSolver() = default;

OP_ERROR SOP_VdbSolver::Cache::cookVDBSop(OP_Context& context) {
	try {
		const fpreal now = context.getTime();

		HoudiniInterrupter boss("Computing VDB grids");

		const GU_Detail* velgeo = inputGeo(1);
		if (!velgeo) {
			addError(SOP_MESSAGE, "No velocity input geometry found");
			return error();
		}

		openvdb::FloatGrid::ConstPtr densGrid;
		openvdb::VectorGrid::ConstPtr velGrid;

		// Get the VDB grids from the input geometry
		for (VdbPrimIterator it(gdp, matchGroup(*gdp, evalStdString("group", now))); it; ++it) {
			if (boss.wasInterrupted()) break;
			densGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>((*it)->getConstGridPtr());
			if (densGrid) break;
		}

		for (VdbPrimCIterator it(velgeo, matchGroup(*velgeo, evalStdString("velgroup", now))); it; ++it) {
			if (boss.wasInterrupted()) break;
			velGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>((*it)->getConstGridPtr());
			if (velGrid) break;
		}

		if (!densGrid || !velGrid) {
			addError(SOP_MESSAGE, "No Valid grids found in the input geometry");
			return error();
		}

		// Process grids:
		const double dt = evalFloat("timestep", 0, now);

		openvdb::FloatGrid::Ptr advectedDens = nullptr;
		openvdb::GridBase::Ptr nanoToOpen = nullptr;

		if (evalInt("GPU", 0, now)) {
			// Convert density and velocity grids if not already converted
			nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle =
			    nanovdb::tools::createNanoGrid<openvdb::FloatGrid, float, nanovdb::cuda::DeviceBuffer>(*densGrid);

			nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> velHandle =
			    nanovdb::tools::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::cuda::DeviceBuffer>(
			        *velGrid);

			// Reuse the same CUDA stream for all operations
			cudaStream_t stream;
			cudaError_t cudaStatus = cudaStreamCreate(&stream);
			if (cudaStatus != cudaSuccess) {
				addError(SOP_MESSAGE, "Failed to create CUDA stream");
				return error();
			}

			// Asynchronous upload of grids to GPU
			handle.deviceUpload(stream, false);
			velHandle.deviceUpload(stream, false);

			// Wait for uploads to complete before launching kernels
			cudaStatus = cudaStreamSynchronize(stream);
			if (cudaStatus != cudaSuccess) {
				addError(SOP_MESSAGE, "CUDA stream synchronization failed");
				return error();
			}

			// Obtain device grid pointers
			nanovdb::FloatGrid* gpuHandle = handle.deviceGrid<float>();
			const nanovdb::Vec3fGrid* velGpuHandle = velHandle.deviceGrid<nanovdb::Vec3f>();

			// Validate grid pointers
			if (!gpuHandle || !velGpuHandle) {
				addError(SOP_MESSAGE, "GridHandle did not contain the expected grid with value type float/Vec3f");
				return error();
			}

			const nanovdb::FloatGrid* cpuHandle = handle.grid<float>();

			// Ensure the grid supports sequential access to leaf nodes
			if (!cpuHandle->isSequential<0>()) {
				addError(SOP_MESSAGE, "Grid does not support sequential access to leaf nodes!");
				return error();
			}

			const auto leafCount = cpuHandle->tree().nodeCount(0);

			// Launch kernels asynchronously
			launch_kernels(gpuHandle, velGpuHandle, leafCount, dt, stream);

			// Asynchronous download of result grid from GPU
			handle.deviceDownload(stream, false);

			// Wait for all GPU tasks to complete
			cudaStatus = cudaStreamSynchronize(stream);
			if (cudaStatus != cudaSuccess) {
				addError(SOP_MESSAGE, "CUDA stream synchronization failed after kernel execution");
				return error();
			}

			// Destroy the CUDA stream
			cudaStatus = cudaStreamDestroy(stream);
			if (cudaStatus != cudaSuccess) {
				addError(SOP_MESSAGE, "Failed to destroy CUDA stream");
				return error();
			}

			// Convert the NanoVDB grid back to OpenVDB
			nanoToOpen = nanovdb::tools::nanoToOpenVDB(handle);

		} else {
			advectedDens = advect<openvdb::FloatGrid>(densGrid, velGrid, dt);

			if (!advectedDens) {
				addError(SOP_MESSAGE, "Failed to advect density grid");
				return error();
			}
		}

		gdp->clearAndDestroy();

		// Create new VDB primitives from the advected grids
		if (evalInt("GPU", 0, now)) {
			const openvdb::FloatGrid::Ptr out = openvdb::gridPtrCast<openvdb::FloatGrid>(nanoToOpen);
			GU_PrimVDB::buildFromGrid(*gdp, out, nullptr, "density");
		} else {
			GU_PrimVDB::buildFromGrid(*gdp, advectedDens, nullptr, "density");
		}


		boss.end();
	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
		return error();
	}

	return error();
}


template <typename GridType>
typename GridType::ConstPtr SOP_VdbSolver::Cache::processGrid(const GridCPtr& in) {
	try {
		typename GridType::ConstPtr transformed = openvdb::gridConstPtrCast<GridType>(in);

		if (!transformed) {
			addError(SOP_MESSAGE, "Input grid is not of the expected type");
			return nullptr;
		}

		return transformed;

	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
		return nullptr;
	}
}

template <typename GridType>
typename GridType::Ptr SOP_VdbSolver::Cache::advect(const openvdb::FloatGrid::ConstPtr& grid,
                                                    const openvdb::VectorGrid::ConstPtr& velocity, const double dt) {
	HoudiniInterrupter boss("Advecting grid...");
	boss.start();

	openvdb::tools::VolumeAdvection<openvdb::VectorGrid> advectOp(*velocity);

	// Use BoxSampler explicitly
	auto result = advectOp.advect<GridType, openvdb::tools::BoxSampler>(*grid, dt);

	boss.end();

	return result;
}