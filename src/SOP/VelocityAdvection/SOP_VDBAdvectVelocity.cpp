#include "SOP_VDBAdvectVelocity.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/GridData.hpp"
#include "Utils/OpenToNano.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"


extern "C" void advect_points_to_grid(const OpenVectorGrid& in_data, NanoVectorGrid& out_data, const float voxelSize, const float dt, const cudaStream_t& stream);


void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvectvelocity", "HNanoAdvectVelocity",
	                                   SOP_HNanoAdvectVelocity::myConstructor,
	                                   SOP_HNanoAdvectVelocity::buildTemplates(), 1, 1, nullptr, OP_FLAG_GENERATOR));
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
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}


void SOP_HNanoAdvectVelocityVerb::cook(const CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_VDBAdvectVelocityParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoAdvectVelocityCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);

	openvdb::VectorGrid::Ptr AGrid = nullptr;

	if (auto err = loadGrid(ageo, AGrid, sopparms.getAgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	OpenVectorGrid open_out_data;
	{
		ScopedTimer timer("Extracting voxels from " + AGrid->getName());
		extractFromOpenVDB<openvdb::VectorGrid, openvdb::Coord, openvdb::Vec3f>(AGrid, open_out_data);
	}

	NanoVectorGrid out_data;
	{
		const auto name = "Creating NanoVDB grids and Compute Advection";
		boss.start(name);
		ScopedTimer timer(name);

		cudaStream_t stream;
		cudaStreamCreate(&stream);
		const float voxelSize = static_cast<float>(AGrid->voxelSize()[0]);
		const float deltaTime = static_cast<float>(sopparms.getTimestep());

		advect_points_to_grid(open_out_data, out_data, voxelSize, deltaTime, stream);

		boss.end();
	}

	{
		ScopedTimer timer("Building Grid " + AGrid->getName());

		const openvdb::VectorGrid::Ptr grid = openvdb::VectorGrid::create();
		grid->setGridClass(openvdb::GRID_STAGGERED);
		grid->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE);
		grid->setTransform(openvdb::math::Transform::createLinearTransform(AGrid->voxelSize()[0]));

		openvdb::tree::ValueAccessor<openvdb::VectorTree> accessor(grid->tree());

		// Can crash due to nullptr in pCoords or pValues
		for (size_t i = 0; i < out_data.size; ++i) {
			const auto& coord = out_data.pCoords[i];
			const auto& value = out_data.pValues[i];

			accessor.setValue(openvdb::Coord(coord.x(), coord.y(), coord.z()),
			                  openvdb::Vec3f(value[0], value[1], value[2]));
		}

		GU_PrimVDB::buildFromGrid(*detail, grid, nullptr, AGrid->getName().c_str());
	}

	delete[] out_data.pCoords;
	delete[] out_data.pValues;
	delete[] open_out_data.pCoords;
	delete[] open_out_data.pValues;
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