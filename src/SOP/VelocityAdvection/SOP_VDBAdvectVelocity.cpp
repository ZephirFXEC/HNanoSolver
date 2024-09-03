#include "SOP_VDBAdvectVelocity.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"


extern "C" void vel_thrust_kernel(nanovdb::Vec3fGrid*, const nanovdb::Vec3fGrid*, uint64_t, float, float, cudaStream_t);


void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvectvelocity", "HNanoAdvectVelocity",
	                                   SOP_HNanoAdvectVelocity::myConstructor,
	                                   SOP_HNanoAdvectVelocity::buildTemplates(), 2, 2, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_HNanoAdvectVelocityVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
		name	"agroup"
		label	"Velocity Volumes Advected"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
    }
    parm {
		name	"bgroup"
		label	"Velocity Volumes Advecting"
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


PRM_Template* SOP_HNanoAdvectVelocity::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBAdvectedVelocity.cpp", SOP_HNanoAdvectVelocityVerb::theDsFile);
	if (templ.justBuilt()) {
		// They don't work, for now all the FloatGrid found in the 1st input will be advected
		// and the velocity field will be the first found in the 2nd input.
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
		templ.setChoiceListPtr("bgroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}


void SOP_HNanoAdvectVelocityVerb::cook(const CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_VDBAdvectVelocityParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoAdvectVelocityCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);
	const GU_Detail* bgeo = cookparms.inputGeo(1);

	openvdb::VectorGrid::ConstPtr AGrid;
	openvdb::VectorGrid::ConstPtr BGrid;

	if (auto err = loadGrid(ageo, AGrid, sopparms.getAgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	if (auto err = loadGrid(bgeo, BGrid, sopparms.getBgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	{
		const auto name = "Creating NanoVDB grids";
		boss.start(name);
		ScopedTimer timer(name);

		sopcache->pAHandle = nanovdb::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::CudaDeviceBuffer>(
		    *BGrid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable, 0);
		sopcache->pBHandle = sopcache->pAHandle.copy<nanovdb::CudaDeviceBuffer>();

		sopcache->pAHandle.deviceUpload(stream, false);
		sopcache->pBHandle.deviceUpload(stream, false);
		boss.end();
	}

	{
		const auto name = "Computing advection";
		boss.start(name);
		ScopedTimer timer(name);

		nanovdb::Vec3fGrid* gpuAGrid = sopcache->pAHandle.deviceGrid<nanovdb::Vec3f>();
		const nanovdb::Vec3fGrid* gpuBGrid = sopcache->pBHandle.deviceGrid<nanovdb::Vec3f>();
		const nanovdb::Vec3fGrid* cpuGrid = sopcache->pAHandle.grid<nanovdb::Vec3f>();

		const uint32_t leafCount = cpuGrid->tree().nodeCount(0);
		const auto voxelSize = static_cast<float>(cpuGrid->voxelSize()[0]);
		const auto deltaTime = static_cast<float>(sopparms.getTimestep());

		vel_thrust_kernel(gpuAGrid, gpuBGrid, leafCount, voxelSize, deltaTime, stream);
		sopcache->pAHandle.deviceDownload(stream, true);

		boss.end();
	}

	{
		const auto name = "Building Grid";
		boss.start(name);
		ScopedTimer timer(name);

		sopcache->pOpenVDBGrid = nanovdb::nanoToOpenVDB(sopcache->pAHandle);
		const openvdb::VectorGrid::Ptr outputGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(sopcache->pOpenVDBGrid);
		GU_PrimVDB::buildFromGrid(*detail, outputGrid, nullptr, AGrid->getName().c_str());

		boss.end();
	}
}


UT_ErrorSeverity SOP_HNanoAdvectVelocityVerb::loadGrid(const GU_Detail* aGeo, openvdb::VectorGrid::ConstPtr& grid,
                                                       const UT_StringHolder& group) {
	ScopedTimer timer("Load input");

	const GA_PrimitiveGroup* groupRef = aGeo->findPrimitiveGroup(group);
	for (openvdb_houdini::VdbPrimCIterator it(aGeo, groupRef); it; ++it) {
		if (const auto vdb = openvdb::gridConstPtrCast<openvdb::VectorGrid>((*it)->getConstGridPtr())) {
			grid = vdb;
			if (grid) break;
		}
	}

	if (!grid) {
		return UT_ERROR_ABORT;
	}

	return UT_ERROR_NONE;
}


const SOP_NodeVerb::Register<SOP_HNanoAdvectVelocityVerb> SOP_HNanoAdvectVelocityVerb::theVerb;
const SOP_NodeVerb* SOP_HNanoAdvectVelocity::cookVerb() const { return SOP_HNanoAdvectVelocityVerb::theVerb.get(); }