#pragma once

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <SOP_VDBAdvectVelocity.proto.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include "SOP_VDBAdvect.proto.h"
#include "Utils/GridData.hpp"
#include "Utils/Utils.hpp"

class SOP_HNanoVDBAdvect final : public SOP_Node {
   public:
	SOP_HNanoVDBAdvect(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);
	}

	~SOP_HNanoVDBAdvect() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
		return new SOP_HNanoVDBAdvect(net, name, op);
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
class SOP_HNanoVDBAdvectCache final : public SOP_NodeCache {
   public:
	SOP_HNanoVDBAdvectCache() : SOP_NodeCache() {}
	~SOP_HNanoVDBAdvectCache() override {
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

extern "C" void pointToGridVectorToDevice(HNS::OpenVectorGrid& in_data, float voxelSize,
                                          nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
                                          const cudaStream_t& stream);

extern "C" void advect_points_to_grid_f(HNS::OpenFloatGrid& in_data, const nanovdb::Vec3fGrid* vel_grid,
                                        HNS::NanoFloatGrid& out_data, float voxelSize, float dt,
                                        const cudaStream_t& stream);