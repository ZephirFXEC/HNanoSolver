//
// Created by zphrfx on 18/10/2024.
//
#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>


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