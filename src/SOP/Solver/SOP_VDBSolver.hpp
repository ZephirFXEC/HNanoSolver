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
#include "Utils/Utils.hpp"

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

	CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_GENERATOR; }

	template <typename GridT>
	[[nodiscard]] static UT_ErrorSeverity loadVDBs(const GU_Detail* aGeo, const GU_Detail* bGeo,
	                                               std::vector<openvdb::GridBase::ConstPtr>& AGrid,
	                                               openvdb::VectorGrid::ConstPtr& BGrid);

	template <typename NanoVDBGridType, typename OpenVDBGridType>
	[[nodiscard]] UT_ErrorSeverity processGrid(const typename OpenVDBGridType::ConstPtr& grid,
	                                           SOP_VdbSolverCache* sopcache, const SOP_VDBSolverParms& sopparms,
	                                           GU_Detail* detail, const cudaStream_t* stream) const;

	template <typename GridT>
	[[nodiscard]] UT_ErrorSeverity convertAndUpload(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& buffer,
	                                                const typename GridT::ConstPtr& grid,
	                                                const cudaStream_t* const stream) const;

	template <typename GridT>
	[[nodiscard]] UT_ErrorSeverity setupGridPointers(SOP_VdbSolverCache* sopcache, GridT* gpuAGrid,
	                                                 const nanovdb::Vec3fGrid* gpuBGrid, const GridT* cpuGrid) const;


	template <typename GridT>
	[[nodiscard]] UT_ErrorSeverity convertToOpenVDBAndBuildGrid(SOP_VdbSolverCache* sopcache, GU_Detail* detail,
	                                                            const std::string& gridName) const;

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;
	static const SOP_NodeVerb::Register<SOP_VdbSolverVerb> theVerb;
	static const char* const theDsFile;
};

template <typename GridT>
struct KernelData {
	GridT* _temp_grid = nullptr;
	GridT* output_grid = nullptr;
	nanovdb::Vec3fGrid* velocity_grid = nullptr;
	int leaf_size = 0;
	float voxel_size = 0.1f;
	float dt = 0;
};

}  // namespace VdbSolver

#endif  // __SOP_VDBSOLVER_HPP__