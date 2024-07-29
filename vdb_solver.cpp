#include <UT/UT_DSOVersion.h> // Mandatory for all DSOs

#include "vdb_solver.hpp"

#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <UT/UT_Interrupt.h>
#include <GU/GU_Detail.h>
#include <OP/OP_AutoLockInputs.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>

#define NANOVDB_USE_OPENVDB
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
#include <GU/GU_PrimVDB.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridHandle.h>

#include "utils.h"

using namespace VdbSolver;

namespace {
	template<template<typename GridT, typename MaskType, typename InterruptT> class ToolT>
	struct ToolOp
	{
		ToolOp(bool t, openvdb::util::NullInterrupter& boss, const openvdb::BoolGrid* mask = nullptr)
			: mMaskGrid(mask)
			, mThreaded(t)
			, mBoss(boss)
		{
		}

		template<typename GridType>
		void operator()(const GridType& inGrid)
		{
			if (mMaskGrid) {

				// match transform
				openvdb::BoolGrid regionMask;
				regionMask.setTransform(inGrid.transform().copy());
				openvdb::tools::resampleToMatch<openvdb::tools::PointSampler>(
					*mMaskGrid, regionMask, mBoss);

				ToolT<GridType, openvdb::BoolGrid, openvdb::util::NullInterrupter> tool(inGrid, regionMask, &mBoss);
				mOutGrid = tool.process(mThreaded);

			}
			else {
				ToolT<GridType, openvdb::BoolGrid, openvdb::util::NullInterrupter> tool(inGrid, &mBoss);
				mOutGrid = tool.process(mThreaded);
			}
		}

		const openvdb::BoolGrid* mMaskGrid;
		openvdb_houdini::GridPtr           mOutGrid;
		bool                    mThreaded;
		openvdb::util::NullInterrupter& mBoss;
	};
}

void newSopOperator(OP_OperatorTable* table) {
	OP_Operator* op;

	op = new OP_Operator(
		"vdbsolver",                      // internal name, needs to be unique in OP_OperatorTable (table containing all nodes for a network type - SOPs in our case, each entry in the table is an object of class OP_Operator which basically defines everything Houdini requires in order to create nodes of the new type)
		"VDB Solver",                     // UI name
		SOP_VdbSolver::myConstructor,     // how to build the node - A class factory function which constructs nodes of this type
		SOP_VdbSolver::myTemplateList,    // my parameters - An array of PRM_Template objects defining the parameters to this operator
		1,                                            // min # of sources
		1);                                           // max # of sources

	// place this operator under the VDB submenu in the TAB menu.
	op->setOpTabSubMenuPath("VDB");

	// after addOperator(), 'table' will take ownership of 'op'
	table->addOperator(op);
}

const char* SOP_VdbSolver::inputLabel(unsigned idx) const {
	switch (idx) {
	case 0: return "Density VDB";
	default: return "default";
	}
}

static PRM_Name debugPRM("debug", "Print debug information");

PRM_Template SOP_VdbSolver::myTemplateList[] = {
	PRM_Template(PRM_TOGGLE, 1, &debugPRM, PRMzeroDefaults),
	PRM_Template()
};

// constructors, destructors, usually there is no need to really modify anything
// here, the constructor's job is to ensure the node is put into the proper
// network
OP_Node* SOP_VdbSolver::myConstructor(OP_Network* net,
	const char* name,
	OP_Operator* op) {
	return new SOP_VdbSolver(net, name, op);
}

SOP_VdbSolver::SOP_VdbSolver(OP_Network* net,
	const char* name,
	OP_Operator* op)
	: SOP_Node(net, name, op) {}

SOP_VdbSolver::~SOP_VdbSolver() {}

OP_ERROR SOP_VdbSolver::cookMySop(OP_Context& context)
{
	// we must lock our inputs before we try to access their geometry, OP_AutoLockInputs will automatically unlock our inputs when we return
	OP_AutoLockInputs inputs(this);
	if (inputs.lock(context) >= UT_ERROR_ABORT)
		return error();

	duplicateSource(0, context);

	openvdb_houdini::HoudiniInterrupter boss(
		(std::string("Computing VDB grids").c_str()));

	const GU_Detail* geo = inputGeo(0);

	openvdb_houdini::GridPtr outGrid;
	const GU_PrimVDB* vdb = nullptr;

	// Iterate over the prim to get the first VDB
	for (GA_Iterator it(geo->getPrimitiveRange()); !it.atEnd(); it.advance())
	{
		if (boss.wasInterrupted()) throw std::runtime_error("Boss was Interupted");

		const GEO_Primitive* prim = geo->getGEOPrimitive(it.getOffset());
		if (dynamic_cast<const GEO_PrimVDB*>(prim))
		{
			vdb = dynamic_cast<const GU_PrimVDB*>(prim);
			break;
		}
	}


	ToolOp<openvdb::tools::MeanCurvature> op(true, boss.interrupter(), nullptr);
	if (openvdb_houdini::GEOvdbApply<openvdb_houdini::NumericGridTypes>(*vdb, op)) {
		outGrid = op.mOutGrid;
	}

	openvdb_houdini::replaceVdbPrimitive(*gdp, outGrid, *const_cast<GU_PrimVDB*>(vdb), true, "density");

	return error();
}

void SOP_VdbSolver::ProcessFloatVDBGrid(const GU_PrimVDB* vdbPrim)
{

	UT_ASSERT(vdbPrim);

	const auto vdbPtrBase = vdbPrim->getConstGridPtr();
	const auto vdbPtr = openvdb::gridConstPtrCast<openvdb::FloatGrid>(vdbPtrBase);

	if (!vdbPtr) {
		addWarning(SOP_MESSAGE, "Skipping non-float VDB grid");
		return;
	}

	// Convert from OpenVDB to NanoVDB, we don't use opentonano because it can't deal with const ptr
	const nanovdb::GridHandle<nanovdb::HostBuffer> cpuHandle = nanovdb::createNanoGrid(*vdbPtr);

	// Get the NanoVDB grid
	const nanovdb::FloatGrid* nanoGrid = cpuHandle.grid<float>();

	if (!nanoGrid)
	{
		addError(SOP_MESSAGE, "Failed to convert to NanoVDB Density grid");
		return;
	}

	if (DEBUG())
	{
		std::cout << nanoGrid->shortGridName() << "\n";
		std::cout << "Density Active Voxel Count : " << nanoGrid->activeVoxelCount() << "\n";
	}
}

