#include "SOP_VDBSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>
#include <UT/UT_Interrupt.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridBuilder.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/VolumeAdvect.h>

using namespace VdbSolver;

static PRM_Name debugPRM("debug", "Print debug information");

void newSopOperator(OP_OperatorTable* table) {
	if (table == nullptr) return;

	houdini_utils::ParmList parms;

	parms.add(houdini_utils::ParmFactory(PRM_TOGGLE, "debug", "Print Debug Information")
	              .setDefault(PRMoneDefaults)
	              .setTooltip("Print debug information to the console")
	              .setDocumentation("Print debug information to the console"));

	// Level set grid
	parms.add(houdini_utils::ParmFactory(PRM_STRING, "group", "Group")
	              .setChoiceList(&houdini_utils::PrimGroupMenuInput1)
	              .setTooltip("VDB grid(s) to advect.")
	              .setDocumentation("A subset of VDBs in the first input to move using the velocity field"
	                                " (see [specifying volumes|/model/volumes#group])"));

	// Velocity grid
	parms.add(houdini_utils::ParmFactory(PRM_STRING, "velgroup", "Velocity")
	              .setChoiceList(&houdini_utils::PrimGroupMenuInput2)
	              .setTooltip("Velocity grid")
	              .setDocumentation("The name of a VDB primitive in the second input to use as"
	                                " the velocity field (see [specifying volumes|/model/volumes#group])\n\n"
	                                "This must be a vector-valued VDB primitive."
	                                " You can use the [Vector Merge node|Node:sop/DW_OpenVDBVectorMerge]"
	                                " to turn a `vel.[xyz]` triple into a single primitive."));

	// Advect: timestep
	parms.add(houdini_utils::ParmFactory(PRM_FLT, "timestep", "Timestep")
	              .setDefault(1, "1.0/$FPS")
	              .setDocumentation("Number of seconds of movement to apply to the input points\n\n"
	                                "The default is `1/$FPS` (one frame's worth of time)."
	                                " You can use negative values to move the points backwards through"
	                                " the velocity field."));


	// Register this operator.
	OpenVDBOpFactory("HNanoAdvect", SOP_VdbSolver::myConstructor, parms, *table)
	    .setNativeName("hnanoadvect")
	    .addInput("VDBs to Advect")
	    .addInput("Velocity VDB")
	    .setVerb(SOP_NodeVerb::COOK_INPLACE, [] { return new SOP_VdbSolver::Cache; });
}

const char* SOP_VdbSolver::inputLabel(const unsigned idx) const {
	switch (idx) {
		case 0:
			return "Input Grids";

		case 1:
			return "Velocity Grids";
		default:
			return "default";
	}
}

OP_Node* SOP_VdbSolver::myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
	return new SOP_VdbSolver(net, name, op);
}

SOP_VdbSolver::SOP_VdbSolver(OP_Network* net, const char* name, OP_Operator* op) : SOP_NodeVDB(net, name, op) {}

SOP_VdbSolver::~SOP_VdbSolver() = default;

OP_ERROR SOP_VdbSolver::Cache::cookVDBSop(OP_Context& context) {
	try {
		const fpreal now = context.getTime();

		HoudiniInterrupter boss("Computing VDB grids");

		const GU_Detail* velgeo = inputGeo(1);
		if (!velgeo) {
			addError(SOP_MESSAGE, "No velocity input geometry found");
			return error();
		}

		const GU_PrimVDB* densvdb = nullptr;
		const GU_PrimVDB* velvdb = nullptr;

		// Get the VDB grids from the input geometry
		for (VdbPrimCIterator it(gdp, matchGroup(*gdp, evalStdString("group", now))); it; ++it) {
			if (boss.wasInterrupted()) break;
			densvdb = *it;
			if (densvdb) break;
		}

		for (VdbPrimCIterator it(velgeo, matchGroup(*velgeo, evalStdString("velgroup", now))); it; ++it) {
			if (boss.wasInterrupted()) break;
			velvdb = *it;
			if (velvdb) break;
		}

		if (!densvdb || !velvdb) {
			addError(SOP_MESSAGE, "No Valid grids found in the input geometry");
		}

		if (evalInt("debug", 0, now)) {
			for (auto iter = densvdb->getMetadata().beginMeta(); iter != densvdb->getMetadata().endMeta(); ++iter) {
				std::cout << iter->first << " = " << iter->second->str() << std::endl;
			}

			for (auto iter	 = velvdb->getMetadata().beginMeta(); iter != velvdb->getMetadata().endMeta(); ++iter) {
				std::cout << iter->first << " = " << iter->second->str() << std::endl;
			}
		}

		// Process the VDB grids
		if (const GridPtr outGrid = processGrid(densvdb->getConstGridPtr(), velvdb->getConstGridPtr(), now)) {
			gdp->clearAndDestroy();
			GU_PrimVDB::buildFromGrid(*gdp, outGrid, densvdb);
		} else {
			addError(SOP_MESSAGE, "Failed to process grids");
		}

		boss.end();

	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
		return error();
	}

	return error();
}

GridPtr SOP_VdbSolver::Cache::processGrid(const GridCPtr& density, const GridCPtr& vel, const float now) {
	try {
		const auto densityFloatGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(density);
		const auto velocityGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>(vel);

		if (!densityFloatGrid) {
			addError(SOP_MESSAGE, "Input density grid is not a FloatGrid");
			return nullptr;
		}

		openvdb::FloatGrid::Ptr outputGrid = densityFloatGrid->deepCopy();

		/*
		constexpr float dt = 0.1f; // Time step for advection, adjust as needed

		openvdb::tools::foreach(outputGrid->beginValueOn(), [&](const openvdb::FloatGrid::ValueOnIter& iter) {
		    const auto& coord = iter.getCoord();
		    const openvdb::Vec3f velocity = velocityGrid->getConstAccessor().getValue(coord);
		    const openvdb::Vec3f advectedPos = coord.asVec3s() - velocity * dt;

		    const float advectedValue = openvdb::tools::BoxSampler::sample(densityFloatGrid->constTree(), advectedPos);
		    iter.setValue(advectedValue);
		});

		*/// ------------

		openvdb::tools::VolumeAdvection<openvdb::VectorGrid, true> advection(*velocityGrid);
		advection.setIntegrator(openvdb::tools::Scheme::SEMI);

		outputGrid = advection.advect<openvdb::FloatGrid, openvdb::tools::BoxSampler>(*densityFloatGrid,
		                                                                              evalFloat("timestep", 0, now));

		return outputGrid;

	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
	}

	return nullptr;
}