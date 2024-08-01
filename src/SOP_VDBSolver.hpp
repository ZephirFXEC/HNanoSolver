#pragma once

#include "OpenVDB_Utils/SOP_NodeVDB.hpp"
#include "OpenVDB_Utils/Utils.hpp"



using namespace openvdb_houdini;
namespace VdbSolver {
class SOP_VdbSolver final : public SOP_NodeVDB {
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
	OP_ERROR cookVDBSop(OP_Context& context) override;

   private:
	// helper function for returning value of parameter
	int DEBUG() const { return evalInt("debug", 0, 0); }

	// helper function for processing VDB primitives
	GridPtr processGrid(const GridCPtr& density, const GridCPtr& vel, UT_AutoInterrupt* boss);
};

}  // namespace VdbSolver
