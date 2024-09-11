#include "SOP_VDBAdvectVelocity.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"


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

	openvdb::VectorGrid::Ptr AGrid = nullptr;
	openvdb::VectorGrid::Ptr BGrid = nullptr;

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

		sopcache->pAHandle.deviceUpload(stream, false);
		boss.end();
	}

	nanovdb::Coord* h_coords = nullptr;
	nanovdb::Vec3f* h_values = nullptr;
	size_t count = 0;
	float voxelSize = 0.5f;

	{
		const auto name = "Computing advection";
		boss.start(name);
		ScopedTimer timer(name);

		nanovdb::Vec3fGrid* gpuAGrid = sopcache->pAHandle.deviceGrid<nanovdb::Vec3f>();
		const nanovdb::Vec3fGrid* cpuGrid = sopcache->pAHandle.grid<nanovdb::Vec3f>();

		const uint32_t leafCount = cpuGrid->tree().nodeCount(0);
		voxelSize = static_cast<float>(cpuGrid->voxelSize()[0]);
		const auto deltaTime = static_cast<float>(sopparms.getTimestep());

		h_coords = new nanovdb::Coord[512 * leafCount];
		h_values = new nanovdb::Vec3f[512 * leafCount];

		vel_thrust_kernel(gpuAGrid, leafCount, voxelSize, deltaTime, stream, h_coords, h_values, count);

		boss.end();
	}

	{
		ScopedTimer timer("Building Grid " + AGrid->getName());

		const openvdb::VectorGrid::Ptr grid = openvdb::VectorGrid::create();
		grid->setGridClass(openvdb::GRID_STAGGERED);
		grid->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE);
		grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));

		auto accessor = grid->getAccessor();

		for (size_t i = 0; i < count; ++i) {
			const auto& coord = h_coords[i];
			const auto& value = h_values[i];

			accessor.setValue(openvdb::Coord(coord.x(), coord.y(), coord.z()),
			                  openvdb::Vec3f(value[0], value[1], value[2]));
		}

		GU_PrimVDB::buildFromGrid(*detail, grid, nullptr, AGrid->getName().c_str());
	}

	delete[] h_coords;
	delete[] h_values;
}


UT_ErrorSeverity SOP_HNanoAdvectVelocityVerb::loadGrid(const GU_Detail* aGeo, openvdb::VectorGrid::Ptr& grid,
                                                       const UT_StringHolder& group) {
	ScopedTimer timer("Load input");

	const GA_PrimitiveGroup* groupRef = aGeo->findPrimitiveGroup(group);
	for (openvdb_houdini::VdbPrimIterator it(aGeo, groupRef); it; ++it) {
		if (const auto vdb = openvdb::gridPtrCast<openvdb::VectorGrid>((*it)->getGridPtr())) {
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