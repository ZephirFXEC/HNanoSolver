//
// Created by zphrfx on 03/10/2024.
//

#include "SOP_VDBProjectNonDivergent.hpp"

#include <UT/UT_DSOVersion.h>


const char* const SOP_HNanoVDBProjectNonDivergentVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
	parm {
		name	"velgrid"
		label	"Velocity Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
	}
	parm {
		name "voxelsize"
		label "Voxel Size"
        type    float
        size    1
        default { "0.5" }
	}
}
)THEDSFILE";

PRM_Template* SOP_HNanoVDBProjectNonDivergent::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBProjectNonDivergent.cpp", SOP_HNanoVDBProjectNonDivergentVerb::theDsFile);
	if (templ.justBuilt()) {
		templ.setChoiceListPtr("velgrid", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoprojectnondivergent", "HNanoProjectNonDivergent",
	                                   SOP_HNanoVDBProjectNonDivergent::myConstructor,
	                                   SOP_HNanoVDBProjectNonDivergent::buildTemplates(), 1, 1, nullptr, 0));
}


const SOP_NodeVerb::Register<SOP_HNanoVDBProjectNonDivergentVerb> SOP_HNanoVDBProjectNonDivergentVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoVDBProjectNonDivergent::cookVerb() const {
	return SOP_HNanoVDBProjectNonDivergentVerb::theVerb.get();
}


void SOP_HNanoVDBProjectNonDivergentVerb::cook(const CookParms& cookparms) const {}