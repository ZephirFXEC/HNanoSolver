#include "SOP_VDBSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

using namespace VdbSolver;

extern "C" void kernel(nanovdb::FloatGrid*, nanovdb::Vec3fGrid*, int, float, float, cudaStream_t stream);
extern "C" void vel_kernel(nanovdb::Vec3fGrid*, const nanovdb::Vec3fGrid*, uint64_t, float, float);
extern "C" void thrust_kernel(nanovdb::FloatGrid*, nanovdb::Vec3fGrid*, int, float, float);
extern "C" void vel_thrust_kernel(nanovdb::Vec3fGrid*, const nanovdb::Vec3fGrid*, uint64_t, float, float);


void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvect", "HNanoAdvect", SOP_VdbSolver::myConstructor,
	                                   SOP_VdbSolver::buildTemplates(), 2, 2, nullptr, 0));
}

const char* const SOP_VdbSolverVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
	parm {
		name "veladvection"
		label "Velocity Advection"
		type toggle
		default { "0" }
	}
    parm {
        name    "advection"
		label	"Advection"
        type    toggle
        default { "1" }
    }
    parm {
		name	"dengroup"
		label	"Density Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
    }
    parm {
		name	"velgroup"
		label	"Velocity Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 1\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
    }
    parm {
        name    "timestep"
        label   "Time Step"
        type    float
        size    1
        default { "1/$FPS" }
    }
}
)THEDSFILE";

PRM_Template* SOP_VdbSolver::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBSolver.cpp", SOP_VdbSolverVerb::theDsFile);
	if (templ.justBuilt()) {
		templ.setChoiceListPtr("dengroup", &SOP_Node::primGroupMenu);
		templ.setChoiceListPtr("velgroup", &SOP_Node::primGroupMenu);
	}
	return templ.templates();
}


const SOP_NodeVerb::Register<SOP_VdbSolverVerb> SOP_VdbSolverVerb::theVerb;

const SOP_NodeVerb* SOP_VdbSolver::cookVerb() const { return SOP_VdbSolverVerb::theVerb.get(); }


void SOP_VdbSolverVerb::cook(const SOP_NodeVerb::CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_VDBSolverParms>();
	const auto sopcache = dynamic_cast<SOP_VdbSolverCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);
	const GU_Detail* bgeo = cookparms.inputGeo(1);

	if (!bgeo || !ageo) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
		return;
	}

	openvdb::GridBase::ConstPtr AGrid;
	openvdb::VectorGrid::ConstPtr BGrid;

	{
		for (openvdb_houdini::VdbPrimCIterator it(ageo); it; ++it) {
			if (boss.wasInterrupted()) break;
			if (sopparms.getVeladvection()) {
				AGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>((*it)->getConstGridPtr());
			} else {
				AGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>((*it)->getConstGridPtr());
			}
			if (AGrid) break;
		}

		if (!sopparms.getVeladvection()) {
			for (openvdb_houdini::VdbPrimCIterator it(bgeo); it; ++it) {
				if (boss.wasInterrupted()) break;
				BGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>((*it)->getConstGridPtr());
				if (BGrid) break;
			}
		}
	}

	if (!AGrid) {
		cookparms.sopAddError(SOP_MESSAGE, "No Valid grids found in the input geometry");
		return;
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	if (!sopparms.getVeladvection()) {
		const auto castedInputGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(AGrid);

		{  // Init class members
			{
				ScopedTimer timer("Convert to NanoVDB and Upload Density to GPU");
				sopcache->pAHandle = nanovdb::createNanoGrid<openvdb::FloatGrid, float, nanovdb::CudaDeviceBuffer>(
				    *castedInputGrid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable, 0);
				sopcache->pAHandle.deviceUpload(stream, false);
			}

			{
				ScopedTimer timer("Convert to NanoVDB and Upload Velocity to GPU");
				sopcache->pBHandle =
				    nanovdb::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::CudaDeviceBuffer>(*BGrid);
				sopcache->pBHandle.deviceUpload(stream, false);
			}
		}

		{
			nanovdb::FloatGrid* gpuAGrid = sopcache->pAHandle.deviceGrid<float>();
			nanovdb::Vec3fGrid* gpuBGrid = sopcache->pBHandle.deviceGrid<nanovdb::Vec3f>();

			const nanovdb::FloatGrid* cpuHandle = sopcache->pAHandle.grid<float>();

			if (!cpuHandle->isSequential<0>()) {
				cookparms.sopAddError(SOP_MESSAGE, "Grid does not support sequential access to leaf nodes!");
				return;
			}

			if (!gpuAGrid || !gpuBGrid) {
				cookparms.sopAddError(SOP_MESSAGE,
				                      "GridHandle did not contain the expected grid with value type float/Vec3f");
				return;
			}

			{
				ScopedTimer timer("Kernel");
				const auto leafCount = cpuHandle->tree().nodeCount(0);
				const auto voxelSize = cpuHandle->voxelSize()[0];
				const double dt = sopparms.getTimestep();
				thrust_kernel(gpuAGrid, gpuBGrid, leafCount, voxelSize, dt);
			}
		}

		{
			ScopedTimer timer("Download from GPU");
			sopcache->pAHandle.deviceDownload(stream, true);
		}
	} else {
		const auto castedInputGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>(AGrid);
		{  // Init class members
			{
				ScopedTimer timer("Convert Velocity Grid to NanoVDB");
				sopcache->pAHandle =
				    nanovdb::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::CudaDeviceBuffer>(
				        *castedInputGrid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable, 0);
				sopcache->pAHandle.deviceUpload(stream, false);
			}

			{
				ScopedTimer timer("Copy Velocity Grid to Temp Grid");
				sopcache->pBHandle = sopcache->pAHandle.copy<nanovdb::CudaDeviceBuffer>();
				sopcache->pBHandle.deviceUpload(stream, false);
			}
		}

		{
			nanovdb::Vec3fGrid* gpuAGrid = sopcache->pAHandle.deviceGrid<nanovdb::Vec3f>();
			nanovdb::Vec3fGrid* gpuBGrid = sopcache->pBHandle.deviceGrid<nanovdb::Vec3f>();

			const nanovdb::Vec3fGrid* cpuHandle = sopcache->pAHandle.grid<nanovdb::Vec3f>();

			if (!cpuHandle->isSequential<0>()) {
				cookparms.sopAddError(SOP_MESSAGE, "Grid does not support sequential access to leaf nodes!");
				return;
			}

			if (!gpuAGrid || !gpuBGrid) {
				cookparms.sopAddError(SOP_MESSAGE,
				                      "GridHandle did not contain the expected grid with value type float/Vec3f");
				return;
			}

			{
				ScopedTimer timer("Kernel");
				const auto leafCount = cpuHandle->tree().nodeCount(0);
				const auto voxelSize = cpuHandle->voxelSize()[0];
				const double dt = sopparms.getTimestep();
				vel_thrust_kernel(gpuAGrid, gpuBGrid, leafCount, voxelSize, dt);
			}
		}

		{
			ScopedTimer timer("Download from GPU");
			sopcache->pAHandle.deviceDownload(stream, true);
		}
	}

	detail->clearAndDestroy();

	{
		ScopedTimer timer("Convert to OpenVDB and Create VDB grid");
		const openvdb::GridBase::Ptr nanoToOpen = nanovdb::nanoToOpenVDB(sopcache->pAHandle);
		if (sopparms.getVeladvection()) {
			const openvdb::VectorGrid::Ptr out = openvdb::gridPtrCast<openvdb::VectorGrid>(nanoToOpen);
			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, "vel");
		} else {
			const openvdb::FloatGrid::Ptr out = openvdb::gridPtrCast<openvdb::FloatGrid>(nanoToOpen);
			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, "density");
		}

	}

	printf("----------------------------------------------------\n");

	boss.end();
}