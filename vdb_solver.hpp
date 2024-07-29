#pragma once

#include <SOP/SOP_Node.h>
#include <GU/GU_PrimVDB.h>

namespace VdbSolver {
	class SOP_VdbSolver : public SOP_Node {
	public:
		// node contructor for HDK
		static OP_Node* myConstructor(OP_Network*, const char*, OP_Operator*);

		// parameter array for Houdini UI
		static PRM_Template myTemplateList[];

	protected:
		// constructor, destructor
		SOP_VdbSolver(OP_Network* net, const char* name, OP_Operator* op);

		virtual ~SOP_VdbSolver();

		// labeling node inputs in Houdini UI
		virtual const char* inputLabel(unsigned idx) const;

		// main function that does geometry processing
		virtual OP_ERROR cookMySop(OP_Context& context);


	private:
		// helper function for returning value of parameter
		int DEBUG() { return evalInt("debug", 0, 0); }

		// helper function for processing VDB primitives
		void ProcessFloatVDBGrid(const GU_PrimVDB* vdbPrim);
		[[maybe_unused]] void ProcessVecVDBGrid(const GU_PrimVDB* vdbPrim);


	};
} // namespace VdbSolver
