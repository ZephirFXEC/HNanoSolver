#include "SOP_HNanoSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <openvdb/tools/VolumeAdvect.h>

#include "Utils/GridBuilder.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvect", "HNanoAdvect", SOP_HNanoSolver::myConstructor,
	                                   SOP_HNanoSolver::buildTemplates(), 2, 2, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_HNanoSolverVerb::theDsFile = R"THEDSFILE(
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
	parm {
		name "voxelsize"
		label "Voxel Size"
        type    float
        size    1
        default { "0.5" }
	}
	parm {
		name "iterations"
		label "Jacobi Iterations"
        type    integer
        size    1
        range   { 1! 100 }
	}
}
)THEDSFILE";


PRM_Template* SOP_HNanoSolver::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_HNanoSolver.cpp", SOP_HNanoSolverVerb::theDsFile);
	if (templ.justBuilt()) {
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
		templ.setChoiceListPtr("bgroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}
const SOP_NodeVerb::Register<SOP_HNanoSolverVerb> SOP_HNanoSolverVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoSolver::cookVerb() const { return SOP_HNanoSolverVerb::theVerb.get(); }

void SOP_HNanoSolverVerb::cook(const CookParms& cookparms) const {


	/*
	 * 1. Merge Input Grids for sourcing
	 * 2. Advect Vel Field
	 * 3. Combustion
	 * 4. Pressure Projection
	 * 5. Advection of Density / Temperature / Fuel ...
	 */


	const auto& sopparms = cookparms.parms<SOP_HNanoSolverParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoSolverCache*>(cookparms.cache());


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


	// Encode data in the Velocity topology


	HNS::OpenVectorGrid vel_out_data;
	{
		ScopedTimer timer("Extracting voxels from " + BGrid[0]->getName());
		HNS::extractFromOpenVDB<openvdb::VectorGrid, openvdb::Vec3f>(BGrid[0], vel_out_data);
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	nanovdb::Vec3fGrid* velGrid;
	{
		ScopedTimer timer("Converting " + BGrid[0]->getName() + " to NanoVDB");
		//pointToGridVectorToDevice(vel_out_data, BGrid[0]->voxelSize()[0], sopcache->pVelHandle, stream);
		velGrid = sopcache->pVelHandle.deviceGrid<nanovdb::Vec3f>();
	}

	std::vector<HNS::OpenFloatGrid> gridData(AGrid.size());
	{
		ScopedTimer timer("Load Grid Data");

		for (size_t i = 0; i < AGrid.size(); ++i) {
			HNS::extractFromOpenVDB<openvdb::FloatGrid, float>(AGrid[i], gridData[i]);
		}
	}

	std::vector<HNS::NanoFloatGrid> advectedData(AGrid.size());
	{
		ScopedTimer timer("Advect Floats");

		const float voxelSize = static_cast<float>(AGrid[0]->voxelSize()[0]);
		const float deltaTime = static_cast<float>(sopparms.getTimestep());

		//AdvectFloats(gridData, velGrid, advectedData, voxelSize, deltaTime, stream);
	}
}
