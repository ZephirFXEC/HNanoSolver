#include "SOP_VDBSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include "SOP_VDBSolver.proto.h"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

using namespace VdbSolver;

extern "C" void kernel(nanovdb::FloatGrid*, nanovdb::Vec3fGrid*, int, float, float, cudaStream_t stream);
extern "C" void thrust_kernel(nanovdb::FloatGrid*, nanovdb::Vec3fGrid*, int, float, float, cudaStream_t stream);

extern "C" void scaleActiveVoxels(nanovdb::FloatGrid*, uint64_t, float);

class SOP_VdbSolverCache final : public SOP_NodeCache {
   public:
	SOP_VdbSolverCache() : SOP_NodeCache() {}
	~SOP_VdbSolverCache() override {
		if(!pDenHandle.isEmpty()) {
			pDenHandle.reset();
		}

		if(!pVelHandle.isEmpty()) {
			pVelHandle.reset();
		}
	}

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> pDenHandle;
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> pVelHandle;
};

class SOP_VdbSolverVerb final : public SOP_NodeVerb {
   public:
	SOP_VdbSolverVerb() = default;
	~SOP_VdbSolverVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VDBSolverParms(); }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_VdbSolverCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "hnanoadvect"; }

	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_DUPLICATE; }

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;

	static const SOP_NodeVerb::Register<SOP_VdbSolverVerb> theVerb;

	static const char* const theDsFile;
};

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvect",
	                                   "HNanoAdvect",
	                                   SOP_VdbSolver::myConstructor,
	                                   SOP_VdbSolver::buildTemplates(),
	                                   2,
	                                   2,
	                                   nullptr,
	                                   0));
}

const char* const SOP_VdbSolverVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
        name    "advection"
		label	"Advection"
        label   "Run the advection kernel"
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
	const GU_Detail* dengeo = cookparms.inputGeo(0);
	const GU_Detail* velgeo = cookparms.inputGeo(1);

	if (!velgeo || !dengeo) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
		return;
	}

	openvdb::FloatGrid::ConstPtr densGrid;
	openvdb::VectorGrid::ConstPtr velGrid;

	{
		for (openvdb_houdini::VdbPrimCIterator it(dengeo); it; ++it) {
			if (boss.wasInterrupted()) break;
			densGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>((*it)->getConstGridPtr());
			if (densGrid) break;
		}

		for (openvdb_houdini::VdbPrimCIterator it(velgeo); it; ++it) {
			if (boss.wasInterrupted()) break;
			velGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>((*it)->getConstGridPtr());
			if (velGrid) break;
		}

		if (!densGrid || !velGrid) {
			cookparms.sopAddError(SOP_MESSAGE, "No Valid grids found in the input geometry");
			return;
		}
	}


	/*
	 * We save in cache the density and velocity grids in NanoVDB format. Each frame the density grid is copied back
	 * on the cpu to create and display the grid in Houdini.
	 * Since this cache isn't cleared when the node is dirtied / bypassed / recooked, we need to merge the sourcing grids
	 * with the cached grid during the sourcing phase.
	 *
	 * TODO: Create a callback to clear the cache.
	 * TODO: Merging grids logic using nanovdb::mergeGrids (cuda)
	 */


	if (sopparms.getGpu()) {

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		{  // Init class members
			{
				ScopedTimer timer("Convert to NanoVDB and Upload Density to GPU");
				sopcache->pDenHandle =
				    nanovdb::createNanoGrid<openvdb::FloatGrid, float, nanovdb::CudaDeviceBuffer>(*densGrid);
				sopcache->pDenHandle.deviceUpload(stream, false);
			}

			{
				ScopedTimer timer("Convert to NanoVDB and Upload Velocity to GPU");
				sopcache->pVelHandle =
				    nanovdb::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::CudaDeviceBuffer>(*velGrid);
				sopcache->pVelHandle.deviceUpload(stream, false);
			}
		}

		{
			nanovdb::FloatGrid* gpuDenGrid = sopcache->pDenHandle.deviceGrid<float>();
			nanovdb::Vec3fGrid* gpuVelGrid = sopcache->pVelHandle.deviceGrid<nanovdb::Vec3f>();

			const nanovdb::FloatGrid* cpuHandle = sopcache->pDenHandle.grid<float>();

			if (!cpuHandle->isSequential<0>()) {
				cookparms.sopAddError(SOP_MESSAGE, "Grid does not support sequential access to leaf nodes!");
				return;
			}

			if (!gpuDenGrid || !gpuVelGrid) {
				cookparms.sopAddError(SOP_MESSAGE,
				                      "GridHandle did not contain the expected grid with value type float/Vec3f");
				return;
			}

			{
				ScopedTimer timer("Kernel");
				const auto leafCount = cpuHandle->tree().nodeCount(0);
				const auto voxelSize = cpuHandle->voxelSize()[0];
				const double dt = sopparms.getTimestep();
				thrust_kernel(gpuDenGrid, gpuVelGrid, leafCount, voxelSize, dt, stream);
			}
		}

		{
			ScopedTimer timer("Download from GPU");
			sopcache->pDenHandle.deviceDownload(stream, true);
		}

		detail->clear();

		{
			ScopedTimer timer("Convert to OpenVDB and Create VDB grid");
			const openvdb::GridBase::Ptr nanoToOpen = nanovdb::nanoToOpenVDB(sopcache->pDenHandle);
			const openvdb::FloatGrid::Ptr out = openvdb::gridPtrCast<openvdb::FloatGrid>(nanoToOpen);
			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, "density");
		}
	}

	boss.end();
}