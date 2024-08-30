//
// Created by zphrfx on 29/08/2024.
//

#include "SOP_ReadWriteTest.hpp"

#include <UT/UT_DSOVersion.h>

#include <vector>

#include "Utils/ScopedTimer.hpp"
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
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GEO_Detail* const in_geo = cookparms.inputGeo(0);

	if (!in_geo) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	std::vector<GU_PrimVDB*> prims;

	for (openvdb_houdini::VdbPrimIterator it(in_geo); it; ++it) {
		if (boss.wasInterrupted()) {
			throw std::runtime_error("processing was interrupted");
		}
		prims.push_back(it.getPrimitive());
		printf("Found VDB prim %s\n", it.getPrimitive()->getGridPtr()->getName().c_str());
	}

	if (prims.empty()) {
		cookparms.sopAddError(SOP_MESSAGE, "First input must contain VDBs!");
	}

	{
		ScopedTimer timer("Making grids unique");
		for (const auto& prim : prims) {
			prim->makeGridUnique();
		}
	}

	std::vector<openvdb::GridBase::Ptr> grids;
	{
		ScopedTimer timer("Extracting grids");
		for (const auto prim : prims) {
			grids.push_back(prim->getGridPtr());
		}
	}

	auto printOp = [](const openvdb::GridBase& in) { in.print(std::cout, 1); };
	for (const auto& grid : grids) {
		grid->apply<openvdb_houdini::VolumeGridTypes>(printOp);
	}
}