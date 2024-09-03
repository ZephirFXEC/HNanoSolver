#pragma once

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include "SOP_VDBAdvectVelocity.proto.h"
#include "Utils/Utils.hpp"

class SOP_HNanoAdvectVelocity final : public SOP_Node {
   public:
	SOP_HNanoAdvectVelocity(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);
	}

	~SOP_HNanoAdvectVelocity() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
		return new SOP_HNanoAdvectVelocity(net, name, op);
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
class SOP_HNanoAdvectVelocityCache final : public SOP_NodeCache {
   public:
	SOP_HNanoAdvectVelocityCache() : SOP_NodeCache() {}
	~SOP_HNanoAdvectVelocityCache() override {
		if (!pAHandle.isEmpty()) {
			pAHandle.reset();
		}

		if (!pBHandle.isEmpty()) {
			pBHandle.reset();
		}
	}

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> pAHandle;
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> pBHandle;
	openvdb::GridBase::Ptr pOpenVDBGrid = nullptr;
};

class SOP_HNanoAdvectVelocityVerb final : public SOP_NodeVerb {
   public:
	SOP_HNanoAdvectVelocityVerb() = default;
	~SOP_HNanoAdvectVelocityVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VDBAdvectVelocityParms(); }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_HNanoAdvectVelocityCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "hnanoadvectvelocity"; }

	CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_GENERATOR; }

	[[nodiscard]] static UT_ErrorSeverity loadGrid(const GU_Detail* aGeo, openvdb::VectorGrid::ConstPtr& grid,
	                                               const UT_StringHolder& group);

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;
	static const SOP_NodeVerb::Register<SOP_HNanoAdvectVelocityVerb> theVerb;
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