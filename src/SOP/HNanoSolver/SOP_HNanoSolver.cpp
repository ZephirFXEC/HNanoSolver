#include "SOP_HNanoSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/OpenToNano.hpp"
#include "Utils/ScopedTimer.hpp"

#include <openvdb/tools/VolumeAdvect.h>

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
}