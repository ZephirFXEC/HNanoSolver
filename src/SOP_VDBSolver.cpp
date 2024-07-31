#include "SOP_VDBSolver.hpp"
#include <UT/UT_DSOVersion.h>


#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <OP/OP_AutoLockInputs.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <UT/UT_Interrupt.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/Stencils.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>

using namespace VdbSolver;

void newSopOperator(OP_OperatorTable* table) {
	auto* op =
	    new OP_Operator("vdbsolver", "VDB Solver", SOP_VdbSolver::myConstructor, SOP_VdbSolver::myTemplateList, 1, 1);

	// place this operator under the VDB submenu in the TAB menu.
	op->setOpTabSubMenuPath("VDB");

	// after addOperator(), 'table' will take ownership of 'op'
	table->addOperator(op);
}

const char* SOP_VdbSolver::inputLabel(unsigned idx) const {
	switch (idx) {
		case 0:
			return "Density VDB";
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
		if (inputs.lock(context) >= UT_ERROR_ABORT) return error();

		duplicateSource(0, context);

		UT_AutoInterrupt boss("Computing VDB grids");

		auto* geo = const_cast<GU_Detail*>(inputGeo(0));
		if (!geo) return error();

		for (VdbPrimIterator it(geo); it; ++it) {
			if (boss.wasInterrupted()) break;

			GU_PrimVDB* vdb = *it;
			if (!vdb) continue;

			GridPtr grid = vdb->getGridPtr();
			if (GridPtr outGrid = processGrid(grid, &boss); outGrid) {
				replaceVdbPrimitive(*gdp, outGrid, *vdb, true, "density");
			}
		}
	} catch (std::exception& e) {
		addError(SOP_MESSAGE, e.what());
		return error();
	}

	return error();
}

GridPtr SOP_VdbSolver::processGrid(const GridPtr& grid, UT_AutoInterrupt* boss) {
	if(grid->isType<openvdb::FloatGrid>()) {
		const auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);

		// Convert OpenVDB grid to NanoVDB grid
		auto handle = nanovdb::openToNanoVDB(floatGrid);

		// Get the NanoVDB grid from the handle
		const auto* nanoGrid = handle.grid<float>();
		if (!nanoGrid) return nullptr;
		auto accessor = nanoGrid->getAccessor();

		// Create a new OpenVDB grid for the output
		auto outGrid = openvdb::FloatGrid::create();
		auto outAccessor = outGrid->getAccessor();

		nanovdb::CurvatureStencil<nanovdb::FloatGrid> CStencil(*nanoGrid);

		const float background = 5.0f;
		const int size = 100;
		auto func = [&](const nanovdb::Coord &ijk){
			float v = 40.0f + 50.0f*(cos(ijk[0]*0.1f)*sin(ijk[1]*0.1f) +
									 cos(ijk[1]*0.1f)*sin(ijk[2]*0.1f) +
									 cos(ijk[2]*0.1f)*sin(ijk[0]*0.1f));
			v = openvdb::math::Max(v, nanovdb::Vec3f(ijk).length() - size);// CSG intersection with a sphere
			return v > background ? background : v < -background ? -background : v;// clamp value
		};

		nanovdb::build::Grid<float> grid(background, "funny", nanovdb::GridClass::FogVolume);
		grid(func, nanovdb::CoordBBox(nanovdb::Coord(-size), nanovdb::Coord(size)));

		auto test = nanovdb::createNanoGrid(grid);

		if(DEBUG()) {
			std::cout << test.gridMetaData()->activeVoxelCount(); // 0
		}

		return nanovdb::nanoToOpenVDB(test);
	}

	return nullptr;
}