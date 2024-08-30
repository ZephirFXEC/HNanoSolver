//
// Created by zphrfx on 29/08/2024.
//

#ifndef SOP_READWRITETEST_HPP
#define SOP_READWRITETEST_HPP

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>

#include "SOP_ReadWriteTest.proto.h"

class SOP_ReadWriteTest final : public SOP_Node {
   public:
	SOP_ReadWriteTest(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);
	}

	~SOP_ReadWriteTest() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) {
		return new SOP_ReadWriteTest(net, name, op);
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

class SOP_ReadWriteTestCache final : public SOP_NodeCache {
   public:
	SOP_ReadWriteTestCache() : SOP_NodeCache() {}
	~SOP_ReadWriteTestCache() override = default;
};

class SOP_ReadWriteTestVerb final : public SOP_NodeVerb {
   public:
	SOP_ReadWriteTestVerb() = default;
	~SOP_ReadWriteTestVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_ReadWriteTestParms; }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_ReadWriteTestCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "HReadWrite"; }

	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_DUPLICATE; }

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;

	static const SOP_NodeVerb::Register<SOP_ReadWriteTestVerb> theVerb;
	static const char* const theDsFile;
};


#endif  // SOP_READWRITETEST_HPP
