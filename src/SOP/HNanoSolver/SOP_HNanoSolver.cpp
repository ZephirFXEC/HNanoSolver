#include "SOP_HNanoSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/VolumeAdvect.h>

#include "../Utils/GridBuilder.hpp"
#include "../Utils/ScopedTimer.hpp"
#include "../Utils/Utils.hpp"
#include "nanovdb/tools/CreateNanoGrid.h"

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanosolver", "HNanoSolver", SOP_HNanoSolver::myConstructor, SOP_HNanoSolver::buildTemplates(), 2,
	                                   2, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_HNanoSolverVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
        name    "timestep"
        label   "Time Step"
        type    float
        size    1
        default { "1/$FPS" }
    }
	parm {
		name "iterations"
		label "Pressure Projection"
        type    integer
        size    1
        range   { 1! 100 }
	}
	parm {
		name "expansion_rate"
		label "Expansion Rate"
		type float
		size 1
		default { "0.1" }
	}
	parm {
		name "temperature_gain"
		label "Temperature Gain"
		type float
		size 1
		default { "0.5" }
	}
	parm {
		name "buoyancy_strength"
		label "Buoyancy Strength"
		type float
		size 1
		default { "1.0" }
	}
	parm {
		name "ambient_temp"
		label "Ambient Temperature"
		type float
		size 1
		default { "23.0" }
	}
	parm {
		name "vorticity"
		label "Vorticity Scale"
		type float
		size 1
		default { "1" }
	}
}
)THEDSFILE";


PRM_Template* SOP_HNanoSolver::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_HNanoSolver.cpp", SOP_HNanoSolverVerb::theDsFile);
	return templ.templates();
}
const SOP_NodeVerb::Register<SOP_HNanoSolverVerb> SOP_HNanoSolverVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoSolver::cookVerb() const { return SOP_HNanoSolverVerb::theVerb.get(); }

void SOP_HNanoSolverVerb::cook(const CookParms& cookparms) const {
	/*
	 * 1. Merge Input Grids for sourcing
	 * 2. Advect Vel Field
	 * 3. Combustion
	 * 4. Pressure Projection
	 * 5. Advection of Density / Temperature / Fuel ...
	 */


	const auto& sopparms = cookparms.parms<SOP_HNanoSolverParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoSolverCache*>(cookparms.cache());


	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* feedback_input = cookparms.inputGeo(0);
	const GU_Detail* source_input = cookparms.inputGeo(1);

	std::vector<openvdb::GridBase::Ptr> feedback_grids;
	std::vector<openvdb::GridBase::Ptr> source_grids;  // Velocity grid ( len = 1 )

	if (auto err = loadGrid(feedback_input, feedback_grids); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load density grid");
		return;
	}

	if (auto err = loadGrid(source_input, source_grids); err != UT_ERROR_NONE && err != UT_ERROR_ABORT) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load density grid");
		return;
	}

	std::vector<openvdb::FloatGrid::Ptr> feedback_float_grids;
	std::vector<openvdb::FloatGrid::Ptr> source_float_grids;
	std::vector<openvdb::VectorGrid::Ptr> feedback_vector_grids;
	std::vector<openvdb::VectorGrid::Ptr> source_vector_grids;

	for (const auto& grid : feedback_grids) {
		if (auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
			feedback_float_grids.push_back(float_grid);
		} else if (auto vector_grid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
			feedback_vector_grids.push_back(vector_grid);
		}
	}

	bool isSourced = !source_grids.empty();

	if (isSourced) {
		ScopedTimer timer("HNanoSolver::Sourcing");
		for (const auto& grid : source_grids) {
			if (auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
				source_float_grids.push_back(float_grid);
			} else if (auto vector_grid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
				source_vector_grids.push_back(vector_grid);
			}
		}
		for (int i = 0; i < source_float_grids.size(); ++i) {
			auto& sourceG = source_float_grids[i];
			auto& feedbackG = feedback_float_grids[i];
			openvdb::tools::compSum(*feedbackG, *sourceG);
		}

		for (int i = 0; i < source_vector_grids.size(); ++i) {
			auto& sourceG = source_vector_grids[i];
			auto& feedbackG = feedback_vector_grids[i];
			openvdb::tools::compSum(*feedbackG, *sourceG);
		}
	}

	// Assume the first vector grid is the primary one for topology/voxel size
	const openvdb::VectorGrid::Ptr primaryVelocityGrid = feedback_vector_grids[0];
	const float voxelSize = static_cast<float>(primaryVelocityGrid->voxelSize()[0]);  // Assuming uniform voxels
	const float deltaTime = static_cast<float>(sopparms.getTimestep());

	openvdb::FloatGrid::Ptr domainGrid = openvdb::FloatGrid::create(0.0f);
	domainGrid->setTransform(primaryVelocityGrid->transform().copy());  // Match transform
	{
		ScopedTimer timer("HNanoSolver::DefineTopology");
		domainGrid->tree().topologyUnion(primaryVelocityGrid->tree());
	}

	HNS::GridIndexedData data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domainGrid, &data);
	builder.setAllocType(AllocationType::Standard);
	{
		for (const auto& grid : feedback_float_grids) {
			builder.addGrid(grid, grid->getName());
		}

		for (const auto& grid : feedback_vector_grids) {
			builder.addGrid(grid, grid->getName());
		}

		builder.build();
	}

	if (data.size() == 0) {
		cookparms.sopAddWarning(SOP_MESSAGE, "No active voxels found. Simulation skipped.");
		return;  // Nothing to simulate
	}

	// --- 4. Create NanoVDB Acceleration Structure ---
	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle;
	{
		ScopedTimer timer("Building NanoVDB Index Grid");
		try {
			CreateIndexGrid(data, handle, voxelSize);
		} catch (const std::exception& e) {
			cookparms.sopAddError(SOP_MESSAGE, (std::string("Failed to create NanoVDB grid: ") + e.what()).c_str());
			return;
		}
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	{
		const int iterations = sopparms.getIterations();

		CombustionParams params{};
		params.expansionRate = sopparms.getExpansion_rate();
		params.temperatureRelease = sopparms.getTemperature_gain();
		params.buoyancyStrength = sopparms.getBuoyancy_strength();
		params.ambientTemp = sopparms.getAmbient_temp();
		params.vortictyScale = sopparms.getVorticity();

		Compute_Sim(data, handle, iterations, deltaTime, primaryVelocityGrid->voxelSize()[0], params, stream);
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	{
		for (const auto& grid : feedback_float_grids) {
			auto out = builder.writeIndexGrid<openvdb::FloatGrid>(grid->getName(), grid->voxelSize()[0]);
			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, out->getName().c_str());
		}

		for (const auto& grid : feedback_vector_grids) {
			auto out = builder.writeIndexGrid<openvdb::VectorGrid>(grid->getName(), grid->voxelSize()[0]);
			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, out->getName().c_str());
		}
	}
}
