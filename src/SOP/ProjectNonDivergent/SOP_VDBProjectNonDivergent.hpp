//
// Created by zphrfx on 03/10/2024.
//

#pragma once

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>

#include "SOP_VDBProjectNonDivergent.proto.h"
#include "Utils/GridData.hpp"

class SOP_HNanoVDBProjectNonDivergent final : public SOP_Node {
   public:
	SOP_HNanoVDBProjectNonDivergent(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);

		// It means we cook at every frame change.
		OP_Node::flags().setTimeDep(true);
	}

	~SOP_HNanoVDBProjectNonDivergent() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
		return new SOP_HNanoVDBProjectNonDivergent(net, name, op);
	}

	OP_ERROR cookMySop(OP_Context& context) override { return cookMyselfAsVerb(context); }

	const SOP_NodeVerb* cookVerb() const override;


	const char* inputLabel(unsigned idx) const override {
		switch (idx) {
			case 0:
				return "Input Grids";
			default:
				return "default";
		}
	}
};

class SOP_HNanoVDBProjectNonDivergentCache final : public SOP_NodeCache {
   public:
	SOP_HNanoVDBProjectNonDivergentCache() : SOP_NodeCache() {}
	~SOP_HNanoVDBProjectNonDivergentCache() override = default;
};

class SOP_HNanoVDBProjectNonDivergentVerb final : public SOP_NodeVerb {
   public:
	SOP_HNanoVDBProjectNonDivergentVerb() = default;
	~SOP_HNanoVDBProjectNonDivergentVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VDBProjectNonDivergentParms; }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_HNanoVDBProjectNonDivergentCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "HNanoProjectNonDivergent"; }

	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_DUPLICATE; }

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;

	static const SOP_NodeVerb::Register<SOP_HNanoVDBProjectNonDivergentVerb> theVerb;
	static const char* const theDsFile;
};


#endif  // SOP_READWRITETEST_HPP
