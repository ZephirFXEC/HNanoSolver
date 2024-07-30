#pragma once

#include <GU/GU_PrimVDB.h>
#include <SOP/SOP_Node.h>

#include "Utils.hpp"


namespace VdbSolver
{
    class SOP_VdbSolver : public SOP_Node
    {
    public:
        // node contructor for HDK
        static OP_Node* myConstructor(OP_Network*, const char*, OP_Operator*);

        // parameter array for Houdini UI
        static PRM_Template myTemplateList[];

    protected:
        // constructor, destructor
        SOP_VdbSolver(OP_Network* net, const char* name, OP_Operator* op);

        ~SOP_VdbSolver() override;

        // labeling node inputs in Houdini UI
        const char* inputLabel(unsigned idx) const override;

        // main function that does geometry processing
        OP_ERROR cookMySop(OP_Context& context) override;

    private:
        // helper function for returning value of parameter
        int DEBUG() { return evalInt("debug", 0, 0); }

        // helper function for processing VDB primitives
        openvdb_houdini::GridPtr ProcessFloatVDBGrid(GU_PrimVDB* vdbPrim, openvdb_houdini::HoudiniInterrupter& boss);
    };
} // namespace VdbSolver
