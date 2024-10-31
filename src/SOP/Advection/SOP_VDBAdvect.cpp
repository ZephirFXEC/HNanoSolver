#include "SOP_VDBAdvect.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/OpenToNano.hpp"
#include "Utils/ScopedTimer.hpp"


void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvect", "HNanoAdvect", SOP_HNanoVDBAdvect::myConstructor,
	                                   SOP_HNanoVDBAdvect::buildTemplates(), 2, 2, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_HNanoVDBAdvectVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
		name	"agroup"
		label	"Density Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
    }
    parm {
		name	"bgroup"
		label	"Velocity Volume"
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


PRM_Template* SOP_HNanoVDBAdvect::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBAdvect.cpp", SOP_HNanoVDBAdvectVerb::theDsFile);
	if (templ.justBuilt()) {
		// They don't work, for now all the FloatGrid found in the 1st input will be advected
		// and the velocity field will be the first found in the 2nd input.
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
		templ.setChoiceListPtr("bgroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}


void SOP_HNanoVDBAdvectVerb::cook(const SOP_NodeVerb::CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_VDBAdvectParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoVDBAdvectCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);
	const GU_Detail* bgeo = cookparms.inputGeo(1);

	std::vector<openvdb::FloatGrid::Ptr> AGrid;
	std::vector<openvdb::VectorGrid::Ptr> BGrid;  // Velocity grid ( len = 1 )

	if (auto err = loadGrid<openvdb::FloatGrid>(ageo, AGrid, sopparms.getAgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load density grid");
	}

	if (auto err = loadGrid<openvdb::VectorGrid>(bgeo, BGrid, sopparms.getBgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load velocity grid");
	}


	{
		boss.start();
		ScopedTimer timer("Total Advection");

		HNS::OpenVectorGrid vel_out_data;
		{
			ScopedTimer timer("Extracting voxels from " + BGrid[0]->getName());
			HNS::extractFromOpenVDB<openvdb::VectorGrid, openvdb::Vec3f>(BGrid[0], vel_out_data);
		}

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		nanovdb::Vec3fGrid* vel_grid;
		{
			ScopedTimer timer("Converting " + BGrid[0]->getName() + " to NanoVDB");
			pointToGridVectorToDevice(vel_out_data, BGrid[0]->voxelSize()[0], sopcache->pBHandle, stream);
			vel_grid = sopcache->pBHandle.deviceGrid<nanovdb::Vec3f>();
		}

		for (auto& grid : AGrid) {
			HNS::OpenFloatGrid open_out_data;
			{
				ScopedTimer timer("Extracting voxels from " + grid->getName());
				HNS::extractFromOpenVDB<openvdb::FloatGrid, float>(grid, open_out_data);
			}

			HNS::NanoFloatGrid out_data;
			{
				const auto name = "Computing " + grid->getName() + " advection";
				boss.start(name.c_str());
				ScopedTimer timer(name);

				const float voxelSize = static_cast<float>(grid->voxelSize()[0]);
				const float deltaTime = static_cast<float>(sopparms.getTimestep());
				advect_points_to_grid_f(open_out_data, vel_grid, out_data, voxelSize, deltaTime, stream);

				boss.end();
			}

			{
				ScopedTimer timer("Building Grid " + grid->getName());

				const openvdb::FloatGrid::Ptr out = openvdb::FloatGrid::create();
				out->setGridClass(openvdb::GRID_FOG_VOLUME);
				out->setTransform(openvdb::math::Transform::createLinearTransform(grid->voxelSize()[0]));

				openvdb::tree::ValueAccessor<openvdb::FloatTree> accessor(out->tree());

				for (size_t i = 0; i < out_data.size; ++i) {
					auto& coord = out_data.pCoords()[i];
					auto value = out_data.pValues()[i];
					accessor.setValue(openvdb::Coord(coord.x(), coord.y(), coord.z()), value);
				}

				GU_PrimVDB::buildFromGrid(*detail, out, nullptr, grid->getName().c_str());
			}
		}

		boss.end();
		cudaStreamDestroy(stream);
	}
}


const SOP_NodeVerb::Register<SOP_HNanoVDBAdvectVerb> SOP_HNanoVDBAdvectVerb::theVerb;
const SOP_NodeVerb* SOP_HNanoVDBAdvect::cookVerb() const { return SOP_HNanoVDBAdvectVerb::theVerb.get(); }