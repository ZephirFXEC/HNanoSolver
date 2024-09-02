#ifndef __SOP_VDBSOLVER_HPP__
#define __SOP_VDBSOLVER_HPP__


#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include "SOP_VDBSolver.proto.h"

namespace VdbSolver {
class SOP_VdbSolver final : public SOP_Node {
   public:
	SOP_VdbSolver(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);
	}

	~SOP_VdbSolver() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
		return new SOP_VdbSolver(net, name, op);
	}

	OP_ERROR cookMySop(OP_Context& context) override { return cookMyselfAsVerb(context); }

	const SOP_NodeVerb* cookVerb() const override;


	const char* inputLabel(unsigned idx) const override {
		switch (idx) {
			case 0:
				return "Input Grids";
			case 1:
				return "Velocity Grids";
			default:
				return "default";
		}
	}
};
class SOP_VdbSolverCache final : public SOP_NodeCache {
   public:
	SOP_VdbSolverCache() : SOP_NodeCache() {}
	~SOP_VdbSolverCache() override {
		if (!pAHandle.isEmpty()) {
			pAHandle.reset();
		}

		if (!pBHandle.isEmpty()) {
			pBHandle.reset();
		}
	}

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> pAHandle;
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> pBHandle;
};

class SOP_VdbSolverVerb final : public SOP_NodeVerb {
   public:
	SOP_VdbSolverVerb() = default;
	~SOP_VdbSolverVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VDBSolverParms(); }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_VdbSolverCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "hnanoadvect"; }

	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_DUPLICATE; }

	template <typename GridT>
	void convertAndUpload(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle, const typename GridT::ConstPtr& grid,
	                      const cudaStream_t* stream) const;

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;

	static const SOP_NodeVerb::Register<SOP_VdbSolverVerb> theVerb;

	static const char* const theDsFile;
};

template <typename GridT>
struct KernelData {
	GridT* _temp_grid;
	GridT* output_grid;
	nanovdb::Vec3fGrid* velocity_grid;
	float voxel_size;
	float dt;
};

}  // namespace VdbSolver

#endif  // __SOP_VDBSOLVER_HPP__