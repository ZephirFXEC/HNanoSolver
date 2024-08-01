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
		houdini_utils::ScopedInputLock lock(*this, context);

		duplicateSource(0, context);
		duplicateSource(1, context);

		UT_AutoInterrupt boss("Computing VDB grids");

		auto* geo = const_cast<GU_Detail*>(inputGeo(0));
		auto* vel = const_cast<GU_Detail*>(inputGeo(1));

		if (!geo && !vel) return error();

		GridPtr densGrid = nullptr, velGrid = nullptr;
		GU_PrimVDB *densvdb = nullptr, *velvdb = nullptr;
		// Get the VDB grids from the input geometry
		{
			for (VdbPrimIterator it(geo); it; ++it) {
				if (boss.wasInterrupted()) break;

				densvdb = *it;
				if (!densvdb) continue;

				densvdb->makeGridUnique();
				densGrid = densvdb->getGridPtr();
			}

			for (VdbPrimIterator it(vel); it; ++it) {
				if (boss.wasInterrupted()) break;

				velvdb = *it;
				if (!velvdb) continue;

				velvdb->makeGridUnique();
				velGrid = velvdb->getGridPtr();
			}
		}

		if (!densGrid || !velGrid) {
			addError(SOP_MESSAGE, "No Valid grids found in the input geometry");
			return error();
		}

		// Process the VDB grids
		{
			if (const GridPtr outGrid = processGrid(densGrid, velGrid, &boss); outGrid) {
				replaceVdbPrimitive(*gdp, outGrid, *densvdb, true, "density");
			}
		}


	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
	}

	return error();
}

GridPtr SOP_VdbSolver::processGrid(const GridPtr& density, const GridPtr& vel, UT_AutoInterrupt* boss) {
	try {
		const openvdb::FloatGrid::ConstPtr densityFloatGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(density);
		const openvdb::VectorGrid::ConstPtr velVectorGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>(vel);

		// if grids don't have the same voxel size then return
		if (densityFloatGrid->voxelSize() != velVectorGrid->voxelSize()) {
			addError(SOP_MESSAGE, "Velocity grid to match Density grid");
			return nullptr;
		}

		openvdb::tools::VolumeAdvection<openvdb::VectorGrid, true> advection(*velVectorGrid);
		advection.setIntegrator(openvdb::tools::Scheme::BFECC);

		const auto advectedGrid =
		    advection.advect<openvdb::FloatGrid, openvdb::tools::PointSampler>(*densityFloatGrid, 0.1);

		if (DEBUG()) {
			std::cout << "Density: " << std::endl;
			densityFloatGrid->print();

			std::cout << "Velocity: " << std::endl;
			velVectorGrid->print();

			std::cout << "Advected: " << std::endl;
			advectedGrid->print();
		}

		return advectedGrid;

	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
	}

	return nullptr;
}