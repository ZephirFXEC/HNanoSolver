#pragma once

#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Utils.hpp"
#include "SOP_VDBAdvect.proto.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/cuda/DeviceBuffer.h"

class SOP_HNanoVDBAdvect final : public SOP_Node {
   public:
	SOP_HNanoVDBAdvect(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) { mySopFlags.setManagesDataIDs(true); }

	~SOP_HNanoVDBAdvect() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) { return new SOP_HNanoVDBAdvect(net, name, op); }

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
class SOP_HNanoVDBAdvectCache final : public SOP_NodeCache {
   public:
	SOP_HNanoVDBAdvectCache() : SOP_NodeCache() {}
	~SOP_HNanoVDBAdvectCache() override {
		if (!pTopologyHandle.isEmpty()) {
			pTopologyHandle.reset();
		}
	}

	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> pTopologyHandle;
};

class SOP_HNanoVDBAdvectVerb final : public SOP_NodeVerb {
   public:
	SOP_HNanoVDBAdvectVerb() = default;
	~SOP_HNanoVDBAdvectVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VDBAdvectParms(); }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_HNanoVDBAdvectCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "hnanoadvect"; }

	CookMode cookMode(const SOP_NodeParms* parms) const override { return COOK_GENERATOR; }


	void cook(const CookParms& cookparms) const override;
	static const Register<SOP_HNanoVDBAdvectVerb> theVerb;
	static const char* const theDsFile;
};

extern "C" void AdvectIndexGrid(HNS::GridIndexedData& data, float dt, float voxelSize, const cudaStream_t& stream);