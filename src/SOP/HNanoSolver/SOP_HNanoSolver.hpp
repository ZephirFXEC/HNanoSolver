/*
 * Created by zphrfx on 25/11/2024.
 *
 * Implement the whole Volumetric solver logic in one node to avoid the overhead of the transfer of the data between nodes.
 * This will allow us to use the GPU to its full potential.
 * The SOP_HNanoSolver will be a single node that will take the source VDBs and output the final advected VDBs.
 */

#pragma once

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>

#include "../Utils/GridData.hpp"
#include "SOP_HNanoSolver.proto.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/cuda/DeviceBuffer.h"


struct CombustionParams {
	float expansionRate;
	float temperatureRelease;
	float buoyancyStrength;
	float ambientTemp;
	float vorticityScale;
	float factorScale;
};

class SOP_HNanoSolver final : public SOP_Node {
   public:
	SOP_HNanoSolver(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);

		// It means we cook at every frame change.
		OP_Node::flags().setTimeDep(true);
	}

	~SOP_HNanoSolver() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) { return new SOP_HNanoSolver(net, name, op); }

	OP_ERROR cookMySop(OP_Context& context) override { return cookMyselfAsVerb(context); }

	const SOP_NodeVerb* cookVerb() const override;


	const char* inputLabel(unsigned idx) const override {
		switch (idx) {
			case 0:
				return "Input Grids";
			default:
				return "Sourcing Grids";
		}
	}
};

class SOP_HNanoSolverCache final : public SOP_NodeCache {
   public:
	SOP_HNanoSolverCache() : SOP_NodeCache() {}
	~SOP_HNanoSolverCache() override = default;
};

class SOP_HNanoSolverVerb final : public SOP_NodeVerb {
   public:
	SOP_HNanoSolverVerb() = default;
	~SOP_HNanoSolverVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_HNanoSolverParms; }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_HNanoSolverCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "HNanoSolver"; }

	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_GENERATOR; }

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;

	static const SOP_NodeVerb::Register<SOP_HNanoSolverVerb> theVerb;
	static const char* const theDsFile;
};

extern "C" void CreateIndexGrid(HNS::GridIndexedData& data, nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, float voxelSize);

extern "C" void Compute_Sim(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, int iteration,
                            float dt, float voxelSize, const CombustionParams& params, bool hasCollision, const cudaStream_t& stream);