#include "SOP_VDBAdvect.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/GridBuilder.hpp"
#include "Utils/ScopedTimer.hpp"

#define NANOVDB_USE_OPENVDB
#include <nanovdb/tools/CreateNanoGrid.h>

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
	std::printf("------------ %s ------------\n", "Begin Advection");

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

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	using SrcGridT = openvdb::FloatGrid;


	SrcGridT::Ptr domain = openvdb::createGrid<openvdb::FloatGrid>();
	{
		ScopedTimer timer("Merging topology");
		domain->topologyUnion(*BGrid[0]);
		domain->tree().voxelizeActiveTiles();
	}

	HNS::GridIndexedData data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &data);
	builder.setAllocType(AllocationType::Standard);
	{
		for (const auto& grid : AGrid) {
			builder.addGrid(grid, grid->getName());
		}

		for (const auto& grid : BGrid) {
			builder.addGrid(grid, grid->getName());
		}

		builder.build();
	}

	{
		ScopedTimer timer_kernel("Launching kernels");
		const float deltaTime = static_cast<float>(sopparms.getTimestep());
		AdvectIndexGrid(data, deltaTime, AGrid[0]->voxelSize()[0], stream);
	}

	{
		openvdb::FloatGrid::Ptr density, temperature, fuel;
		tbb::parallel_invoke([&] { density = builder.writeIndexGrid<openvdb::FloatGrid>("density", AGrid[0]->voxelSize()[0]); },
		                     [&] { temperature = builder.writeIndexGrid<openvdb::FloatGrid>("temperature", AGrid[0]->voxelSize()[0]); },
		                     [&] { fuel = builder.writeIndexGrid<openvdb::FloatGrid>("fuel", AGrid[0]->voxelSize()[0]); });

		GU_PrimVDB::buildFromGrid(*detail, density, nullptr, density->getName().c_str());
		GU_PrimVDB::buildFromGrid(*detail, temperature, nullptr, temperature->getName().c_str());
		GU_PrimVDB::buildFromGrid(*detail, fuel, nullptr, fuel->getName().c_str());
	}

	cudaStreamDestroy(stream);

	std::printf("------------ %s ------------\n", "End Advection");
}


const SOP_NodeVerb::Register<SOP_HNanoVDBAdvectVerb> SOP_HNanoVDBAdvectVerb::theVerb;
const SOP_NodeVerb* SOP_HNanoVDBAdvect::cookVerb() const { return SOP_HNanoVDBAdvectVerb::theVerb.get(); }