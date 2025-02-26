#pragma once

#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Utils.hpp"
#include "SOP_VDBAdvectVelocity.proto.h"
#include "nanovdb/cuda/DeviceBuffer.h"

class SOP_HNanoAdvectVelocity final : public SOP_Node {
   public:
	SOP_HNanoAdvectVelocity(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);
	}

	~SOP_HNanoAdvectVelocity() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) { return new SOP_HNanoAdvectVelocity(net, name, op); }

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
	~SOP_HNanoAdvectVelocityCache() override = default;
};

class SOP_HNanoAdvectVelocityVerb final : public SOP_NodeVerb {
   public:
	SOP_HNanoAdvectVelocityVerb() = default;
	~SOP_HNanoAdvectVelocityVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VDBAdvectVelocityParms(); }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_HNanoAdvectVelocityCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "hnanoadvectvelocity"; }

	CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_GENERATOR; }

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;
	static const SOP_NodeVerb::Register<SOP_HNanoAdvectVelocityVerb> theVerb;
	static const char* const theDsFile;
};

extern "C" void AdvectIndexGridVelocity(HNS::GridIndexedData& data, float dt, float voxelSize, const cudaStream_t& stream);