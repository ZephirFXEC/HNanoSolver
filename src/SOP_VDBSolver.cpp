#include "SOP_VDBSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <OP/OP_AutoLockInputs.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <UT/UT_DSOVersion.h>
#include <UT/UT_Interrupt.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridBuilder.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/VolumeAdvect.h>

using namespace VdbSolver;

void newSopOperator(OP_OperatorTable* table) {
	auto* op =
	    new OP_Operator("vdbsolver", "VDB Solver", SOP_VdbSolver::myConstructor, SOP_VdbSolver::myTemplateList, 2, 2);

	// place this operator under the VDB submenu in the TAB menu.
	op->setOpTabSubMenuPath("VDB");

	// after addOperator(), 'table' will take ownership of 'op'
	table->addOperator(op);
}

const char* SOP_VdbSolver::inputLabel(unsigned idx) const {
	switch (idx) {
		case 0:
			return "Density VDB";
		case 1:
			return "Velocity VDB";
		default:
			return "default";
	}
}

static PRM_Name debugPRM("debug", "Print debug information");

PRM_Template SOP_VdbSolver::myTemplateList[] = {PRM_Template(PRM_TOGGLE, 1, &debugPRM, PRMzeroDefaults),
                                                PRM_Template()};

OP_Node* SOP_VdbSolver::myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
	return new SOP_VdbSolver(net, name, op);
}

SOP_VdbSolver::SOP_VdbSolver(OP_Network* net, const char* name, OP_Operator* op) : SOP_NodeVDB(net, name, op) {}

SOP_VdbSolver::~SOP_VdbSolver() = default;

OP_ERROR SOP_VdbSolver::cookVDBSop(OP_Context& context) {
	try {
		OP_AutoLockInputs inputs(this);
		if (inputs.lock(context) >= UT_ERROR_ABORT)
			return error();

		duplicateSource(0, context);
		duplicateSource(1, context);

		UT_AutoInterrupt boss("Computing VDB grids");

		const GU_Detail* geo = inputGeo(0);
		const GU_Detail* vel = inputGeo(1);

		if (!geo || !vel) return error();

		const GU_PrimVDB* densvdb = nullptr;
		const GU_PrimVDB* velvdb = nullptr;

		// Get the VDB grids from the input geometry
		for (VdbPrimCIterator it(geo); it; ++it) {
			if (boss.wasInterrupted()) break;
			densvdb = *it;
			if (densvdb) break;
		}

		for (VdbPrimCIterator it(vel); it; ++it) {
			if (boss.wasInterrupted()) break;
			velvdb = *it;
			if (velvdb) break;
		}

		if (!densvdb || !velvdb) {
			addError(SOP_MESSAGE, "No Valid grids found in the input geometry");
			return error();
		}

		// Process the VDB grids
		if (const GridPtr outGrid = processGrid(densvdb->getConstGridPtr(), velvdb->getConstGridPtr(), &boss)) {
			gdp->clearAndDestroy();
			GU_PrimVDB::buildFromGrid(*gdp, outGrid, densvdb);
		}

	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
	}

	return error();
}

GridPtr SOP_VdbSolver::processGrid(const GridCPtr& density, const GridCPtr& vel, UT_AutoInterrupt* boss) {
	try {
		const auto densityFloatGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(density);
		const auto velocityGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>(vel);

		if (!densityFloatGrid) {
			addError(SOP_MESSAGE, "Input density grid is not a FloatGrid");
			return nullptr;
		}

		openvdb::FloatGrid::Ptr outputGrid = densityFloatGrid->deepCopy();

		float dt = 0.1f; // Time step for advection, adjust as needed

		openvdb::tools::foreach(outputGrid->beginValueOn(), [&](const openvdb::FloatGrid::ValueOnIter& iter) {
			const auto& coord = iter.getCoord();
			const openvdb::Vec3f velocity = velocityGrid->getConstAccessor().getValue(coord);
			const openvdb::Vec3f advectedPos = coord.asVec3s() - velocity * dt;

			const float advectedValue = openvdb::tools::BoxSampler::sample(densityFloatGrid->tree(), advectedPos);
			iter.setValue(advectedValue);
		});

		if (DEBUG()) {
			for (auto iter = outputGrid->beginMeta(); iter != outputGrid->endMeta(); ++iter) {
				std::cout << iter->first << " = " << iter->second->str() << std::endl;
			}
		}

		return outputGrid;

	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
	}

	return nullptr;
}