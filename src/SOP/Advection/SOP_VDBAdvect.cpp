#include "SOP_VDBAdvect.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/GridBuilder.hpp"
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

		std::vector<HNS::NanoFloatGrid> out_data(AGrid.size());
		std::vector<HNS::OpenFloatGrid> open_out_data(AGrid.size());
		std::vector<cudaStream_t> streams(AGrid.size());
		for (size_t i = 0; i < AGrid.size(); ++i) {
			cudaStream_t s;
			cudaStreamCreate(&s);
			streams[i] = s;
		}

		for (size_t i = 0; i < AGrid.size(); ++i) {
			ScopedTimer timer("Extracting voxels from " + AGrid[i]->getName());
			HNS::extractFromOpenVDB<openvdb::FloatGrid, float>(AGrid[i], open_out_data[i]);
		}

		for (size_t i = 0; i < AGrid.size(); ++i) {
			const auto name = "Computing " + AGrid[i]->getName() + " advection";
			ScopedTimer timer(name);

			const float voxelSize = static_cast<float>(AGrid[i]->voxelSize()[0]);
			const float deltaTime = static_cast<float>(sopparms.getTimestep());
			cudaStreamSynchronize(stream);
			AdvectFloat(open_out_data[i], vel_grid, out_data[i], voxelSize, deltaTime, streams[i]);
		}

		for (size_t i = 0; i < AGrid.size(); ++i) {
			ScopedTimer timer("Building Grid " + AGrid[i]->getName());

			const openvdb::FloatGrid::Ptr out = openvdb::FloatGrid::create();
			out->setGridClass(openvdb::GRID_FOG_VOLUME);
			out->setTransform(openvdb::math::Transform::createLinearTransform(AGrid[i]->voxelSize()[0]));

			openvdb::tree::ValueAccessor<openvdb::FloatTree, false> accessor(out->tree());

			for (size_t j = 0; j < out_data[i].size; ++j) {
				auto& coord = out_data[i].pCoords()[j];
				auto value = out_data[i].pValues()[j];
				accessor.setValue(openvdb::Coord(coord.x(), coord.y(), coord.z()), value);
			}

			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, AGrid[i]->getName().c_str());
		}

		cudaStreamDestroy(stream);
		for (auto s : streams) {
			cudaStreamDestroy(s);
		}
	}

}


const SOP_NodeVerb::Register<SOP_HNanoVDBAdvectVerb> SOP_HNanoVDBAdvectVerb::theVerb;
const SOP_NodeVerb* SOP_HNanoVDBAdvect::cookVerb() const { return SOP_HNanoVDBAdvectVerb::theVerb.get(); }