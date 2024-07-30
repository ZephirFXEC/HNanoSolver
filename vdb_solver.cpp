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

#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
#include <GU/GU_PrimVDB.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridHandle.h>

using namespace VdbSolver;

namespace
{
    template <template<typename GridT, typename MaskType, typename InterruptT> class ToolT>
    struct ToolOp
    {
        ToolOp(bool t, openvdb::util::NullInterrupter& boss, const openvdb::BoolGrid* mask = nullptr)
            : mMaskGrid(mask)
              , mThreaded(t)
              , mBoss(boss)
        {
        }

        template <typename GridType>
        void operator()(const GridType& inGrid)
        {
            if (mMaskGrid)
            {
                // match transform
                openvdb::BoolGrid regionMask;
                regionMask.setTransform(inGrid.transform().copy());
                openvdb::tools::resampleToMatch<openvdb::tools::PointSampler>(
                    *mMaskGrid, regionMask, mBoss);

                ToolT<GridType, openvdb::BoolGrid, openvdb::util::NullInterrupter> tool(inGrid, regionMask, &mBoss);
                mOutGrid = tool.process(mThreaded);
            }
            else
            {
                ToolT<GridType, openvdb::BoolGrid, openvdb::util::NullInterrupter> tool(inGrid, &mBoss);
                mOutGrid = tool.process(mThreaded);
            }
        }

        const openvdb::BoolGrid* mMaskGrid;
        openvdb_houdini::GridPtr mOutGrid;
        bool mThreaded;
        openvdb::util::NullInterrupter& mBoss;
    };
}

void newSopOperator(OP_OperatorTable* table)
{
    auto* op = new OP_Operator(
        "vdbsolver",
        "VDB Solver",
        SOP_VdbSolver::myConstructor,
        SOP_VdbSolver::myTemplateList,
        1,
        1);

    // place this operator under the VDB submenu in the TAB menu.
    op->setOpTabSubMenuPath("VDB");

    // after addOperator(), 'table' will take ownership of 'op'
    table->addOperator(op);
}

const char* SOP_VdbSolver::inputLabel(unsigned idx) const
{
    switch (idx)
    {
    case 0: return "Density VDB";
    default: return "default";
    }
}

static PRM_Name debugPRM("debug", "Print debug information");

PRM_Template SOP_VdbSolver::myTemplateList[] = {
    PRM_Template(PRM_TOGGLE, 1, &debugPRM, PRMzeroDefaults),
    PRM_Template()
};

OP_Node* SOP_VdbSolver::myConstructor(OP_Network* net,
                                      const char* name,
                                      OP_Operator* op)
{
    return new SOP_VdbSolver(net, name, op);
}

SOP_VdbSolver::SOP_VdbSolver(OP_Network* net,
                             const char* name,
                             OP_Operator* op)
    : SOP_Node(net, name, op)
{
}

SOP_VdbSolver::~SOP_VdbSolver() = default;

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

    GU_PrimVDB* vdb = nullptr;

    // Iterate over the prim to get the first VDB
    for (GA_Iterator it(geo->getPrimitiveRange()); !it.atEnd(); it.advance())
    {
        if (boss.wasInterrupted()) throw std::runtime_error("Boss was Interupted");

        if (auto* prim = const_cast<GEO_Primitive*>(geo->getGEOPrimitive(it.getOffset()));
            dynamic_cast<GEO_PrimVDB*>(prim))
        {
            vdb = dynamic_cast<GU_PrimVDB*>(prim);
            break;
        }
    }

    const openvdb_houdini::GridPtr outGrid = ProcessFloatVDBGrid(vdb, boss);

    openvdb_houdini::replaceVdbPrimitive(*gdp, outGrid, *vdb, true, "density");

    return error();
}

openvdb_houdini::GridPtr SOP_VdbSolver::ProcessFloatVDBGrid(GU_PrimVDB* vdbPrim,
                                                            openvdb_houdini::HoudiniInterrupter& boss)
{
    UT_ASSERT(vdbPrim);

    auto vdbPtrBase = vdbPrim->getGridPtr();

    openvdb_houdini::GridPtr outGrid = nullptr;
    ToolOp<openvdb::tools::MeanCurvature> op(true, boss.interrupter(), nullptr);
    if (openvdb_houdini::GEOvdbApply<openvdb_houdini::NumericGridTypes>(vdbPtrBase, op))
    {
        outGrid = op.mOutGrid;
    }

    const auto vdbPtr = openvdb::gridConstPtrCast<openvdb::FloatGrid>(outGrid);


    if (!vdbPtr)
    {
        addWarning(SOP_MESSAGE, "Skipping non-float VDB grid");
        return nullptr;
    }

    // Convert from OpenVDB to NanoVDB, we don't use opentonano because it can't deal with const ptr
    const nanovdb::GridHandle<nanovdb::HostBuffer> cpuHandle = nanovdb::createNanoGrid(*vdbPtr);

    // Get the NanoVDB grid
    const nanovdb::FloatGrid* nanoGrid = cpuHandle.grid<float>();

    if (!nanoGrid)
    {
        addError(SOP_MESSAGE, "Failed to convert to NanoVDB Density grid");
        return nullptr;
    }

    if (DEBUG())
    {
        std::cout << nanoGrid->shortGridName() << "\n";
        std::cout << "Density Active Voxel Count : " << nanoGrid->activeVoxelCount() << "\n";
    }

    return outGrid;
}
