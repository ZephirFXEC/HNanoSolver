//
// Created by zphrfx on 29/08/2024.
//

#include "SOP_ReadWriteTest.hpp"

#include <UT/UT_DSOVersion.h>

#include "Utils/Utils.hpp"

const char* const SOP_ReadWriteTestVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
}
)THEDSFILE";

PRM_Template* SOP_ReadWriteTest::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBSolver.cpp", SOP_ReadWriteTestVerb::theDsFile);
	return templ.templates();
}

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hreadwrite", "HReadWrite", SOP_ReadWriteTest::myConstructor,
	                                   SOP_ReadWriteTest::buildTemplates(), 1, 1, nullptr, 0));
}


const SOP_NodeVerb::Register<SOP_ReadWriteTestVerb> SOP_ReadWriteTestVerb::theVerb;

const SOP_NodeVerb* SOP_ReadWriteTest::cookVerb() const { return SOP_ReadWriteTestVerb::theVerb.get(); }


void SOP_ReadWriteTestVerb::cook(const SOP_NodeVerb::CookParms& cookparms) const {
	const auto sopcache = dynamic_cast<SOP_ReadWriteTestCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);

	if (!ageo) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}


	boss.end();
}