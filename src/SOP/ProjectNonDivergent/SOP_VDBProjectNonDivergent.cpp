//
// Created by zphrfx on 03/10/2024.
//

#include "SOP_VDBProjectNonDivergent.hpp"

#include <UT/UT_DSOVersion.h>
#include <nanovdb/NanoVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>

#include <Utils/Utils.hpp>
#include <utility>


const char* const SOP_HNanoVDBProjectNonDivergentVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
	parm {
		name	"velgrid"
		label	"Velocity Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
	}
	parm {
		name "voxelsize"
		label "Voxel Size"
        type    float
        size    1
        default { "0.5" }
	}
}
)THEDSFILE";

PRM_Template* SOP_HNanoVDBProjectNonDivergent::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBProjectNonDivergent.cpp", SOP_HNanoVDBProjectNonDivergentVerb::theDsFile);
	if (templ.justBuilt()) {
		templ.setChoiceListPtr("velgrid", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoprojectnondivergent", "HNanoProjectNonDivergent",
	                                   SOP_HNanoVDBProjectNonDivergent::myConstructor,
	                                   SOP_HNanoVDBProjectNonDivergent::buildTemplates(), 1, 1, nullptr, 0));
}


const SOP_NodeVerb::Register<SOP_HNanoVDBProjectNonDivergentVerb> SOP_HNanoVDBProjectNonDivergentVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoVDBProjectNonDivergent::cookVerb() const {
	return SOP_HNanoVDBProjectNonDivergentVerb::theVerb.get();
}


void SOP_HNanoVDBProjectNonDivergentVerb::cook(const CookParms& cookparms) const {
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");
	const auto& sopparms = cookparms.parms<SOP_VDBProjectNonDivergentParms>();
	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* in_geo = cookparms.inputGeo(0);

	openvdb::VectorGrid::ConstPtr in_velocity = nullptr;
	for (openvdb_houdini::VdbPrimIterator it(in_geo); it; ++it) {
		if (const auto vdb = openvdb::gridPtrCast<openvdb::VectorGrid>((*it)->getGridPtr())) {
			in_velocity = vdb;
		}
	}
	if (in_velocity == nullptr) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	openvdb::FloatGrid::Ptr pressureGrid = openvdb::FloatGrid::create();
	openvdb::FloatGrid::Ptr divergenceGrid = openvdb::FloatGrid::create();
}


struct MultigridSolver {
	void calculateDivergence(const openvdb::VectorGrid::ConstPtr& velocityGrid,
	                         openvdb::FloatGrid::Ptr& divergenceGrid) {
		divergenceGrid = openvdb::tools::divergence(*velocityGrid);
	}

	void clearPressure(const openvdb::FloatGrid::Ptr& pressureGrid) { pressureGrid->clear(); }


	void calcStencilPressure(const openvdb::Coord& coord, const float omega,
	                         openvdb::FloatGrid::Accessor& pressureAccessor,
	                         const openvdb::FloatGrid::ConstAccessor& divergenceAccessor) {
		const float pE = pressureAccessor.getValue(coord.offsetBy(1, 0, 0));
		const float pW = pressureAccessor.getValue(coord.offsetBy(-1, 0, 0));
		const float pN = pressureAccessor.getValue(coord.offsetBy(0, 1, 0));
		const float pS = pressureAccessor.getValue(coord.offsetBy(0, -1, 0));
		const float pT = pressureAccessor.getValue(coord.offsetBy(0, 0, 1));
		const float pB = pressureAccessor.getValue(coord.offsetBy(0, 0, -1));

		const float rhs = divergenceAccessor.getValue(coord);
		const float p = (rhs + pW + pE + pS + pN + pT + pB) / 6.0f;

		const float currentP = pressureAccessor.getValue(coord);
		pressureAccessor.setValue(coord, (1 - omega) * currentP + omega * p);
	}

	void calcIterate(const int iterations, const float omega, const openvdb::FloatGrid::Ptr& pressureGrid,
	                 const openvdb::FloatGrid::Ptr& divergenceGrid) {
		openvdb::FloatGrid::Accessor pressureAccessor = pressureGrid->getAccessor();
		const openvdb::FloatGrid::ConstAccessor divergenceAccessor = divergenceGrid->getConstAccessor();

		for (int n = 0; n < iterations; ++n) {
			for (auto iter = pressureGrid->beginValueOn(); iter; ++iter) {
				calcStencilPressure(iter.getCoord(), omega, pressureAccessor, divergenceAccessor);
			}
		}
	}

	void calcResidual(const openvdb::FloatGrid::Ptr& pressureGrid, const openvdb::FloatGrid::Ptr& residualGrid,
	                  const openvdb::FloatGrid::Ptr& divergenceGrid) {
		const auto pressureAccessor = pressureGrid->getConstAccessor();
		const auto divergenceAccessor = divergenceGrid->getConstAccessor();
		auto residualAccessor = residualGrid->getAccessor();

		for (auto iter = pressureGrid->cbeginValueOn(); iter; ++iter) {
			const openvdb::Coord xyz = iter.getCoord();
			const float pE = pressureAccessor.getValue(xyz.offsetBy(1, 0, 0));
			const float pW = pressureAccessor.getValue(xyz.offsetBy(-1, 0, 0));
			const float pN = pressureAccessor.getValue(xyz.offsetBy(0, 1, 0));
			const float pS = pressureAccessor.getValue(xyz.offsetBy(0, -1, 0));
			const float pT = pressureAccessor.getValue(xyz.offsetBy(0, 0, 1));
			const float pB = pressureAccessor.getValue(xyz.offsetBy(0, 0, -1));
			const float pC = iter.getValue();

			const float residual = divergenceAccessor.getValue(xyz) + pW + pE + pS + pN + pT + pB - 6 * pC;
			residualAccessor.setValue(xyz, residual);
		}
	}

	void applyPressureGradient(const openvdb::FloatGrid::Ptr& pressureGrid, openvdb::VectorGrid::Ptr& velocityGrid) {
		const openvdb::Vec3SGrid::Ptr gradientGrid = openvdb::tools::gradient(*pressureGrid);

		auto velocityAccessor = velocityGrid->getAccessor();
		for (auto iter = gradientGrid->cbeginValueOn(); iter; ++iter) {
			const openvdb::Coord xyz = iter.getCoord();
			const openvdb::Vec3f gradient = iter.getValue();
			const openvdb::Vec3f velocity = velocityAccessor.getValue(xyz);

			velocityAccessor.setValue(xyz, velocity - gradient);
		}
	}

	void restriction(const openvdb::FloatGrid::Ptr& fineGrid, openvdb::FloatGrid::Ptr& coarseGrid) {
		coarseGrid = openvdb::FloatGrid::create();
		coarseGrid->setTransform(openvdb::math::Transform::createLinearTransform(fineGrid->voxelSize()[0] * 2.0f));
		openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*fineGrid, *coarseGrid);
	}

	void prolongation(const openvdb::FloatGrid::Ptr& coarseGrid, openvdb::FloatGrid::Ptr& fineGrid) {
		fineGrid = openvdb::FloatGrid::create();
		fineGrid->setTransform(openvdb::math::Transform::createLinearTransform(coarseGrid->voxelSize()[0] * 0.5f));
		openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*coarseGrid, *fineGrid);

		for (auto iter = fineGrid->beginValueOn(); iter; ++iter) {
			iter.setValue(iter.getValue() * 8.0f);
		}
	}


	void VCycle(const int maxLevels, const int preSmooth, const int postSmooth,
	            std::vector<openvdb::FloatGrid::Ptr>& pressureHierarchy,
	            std::vector<openvdb::FloatGrid::Ptr>& residualHierarchy) {

		pressureHierarchy.resize(maxLevels);
		residualHierarchy.resize(maxLevels);

		// Restriction phase
		for (int level = 1; level < maxLevels; ++level) {
			restriction(pressureHierarchy[level - 1], pressureHierarchy[level]);
			restriction(residualHierarchy[level - 1], residualHierarchy[level]);
		}

		// Solve on all levels
		for (int level = 0; level < maxLevels; ++level) {
			const int smoothSteps = (level == maxLevels - 1) ? 50 : preSmooth;
			calcIterate(smoothSteps, 1.0f, pressureHierarchy[level], residualHierarchy[level]);

			if (level < maxLevels - 1) {
				calcResidual(pressureHierarchy[level], residualHierarchy[level], residualHierarchy[level]);
				clearPressure(pressureHierarchy[level + 1]);
			}
		}

		// Prolongation phase
		for (int level = maxLevels - 2; level >= 0; --level) {
			prolongation(pressureHierarchy[level + 1], pressureHierarchy[level]);
			calcIterate(postSmooth, 1.0f, pressureHierarchy[level], residualHierarchy[level]);
		}
	}
};