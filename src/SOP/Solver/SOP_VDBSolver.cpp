#include "SOP_VDBSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"


using namespace VdbSolver;

extern "C" void thrust_kernel(const nanovdb::FloatGrid*, nanovdb::FloatGrid*, const nanovdb::Vec3fGrid*, int, float,
                              float, cudaStream_t);

extern "C" void vel_thrust_kernel(nanovdb::Vec3fGrid*, const nanovdb::Vec3fGrid*, uint64_t, float, float, cudaStream_t);


void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvect", "HNanoAdvect", SOP_VdbSolver::myConstructor,
	                                   SOP_VdbSolver::buildTemplates(), 2, 2, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_VdbSolverVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
	parm {
		name "veladvection"
		label "Velocity Advection"
		type toggle
		default { "0" }
	}
    parm {
        name    "advection"
		label	"Advection"
        type    toggle
        default { "1" }
    }
    parm {
		name	"dengroup"
		label	"Density Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
    }
    parm {
		name	"velgroup"
		label	"Velocity Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 1\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
    }
    parm {
        name    "timestep"
        label   "Time Step"
        type    float
        size    1
        default { "1/$FPS" }
    }
}
)THEDSFILE";


PRM_Template* SOP_VdbSolver::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBSolver.cpp", SOP_VdbSolverVerb::theDsFile);
	if (templ.justBuilt()) {
		// They don't work, for now all the FloatGrid found in the 1st input will be advected
		// and the velocity field will be the first found in the 2nd input.
		templ.setChoiceListPtr("dengroup", &SOP_Node::primNamedGroupMenu);
		templ.setChoiceListPtr("velgroup", &SOP_Node::primNamedGroupMenu);
	}
	return templ.templates();
}


void SOP_VdbSolverVerb::cook(const SOP_NodeVerb::CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_VDBSolverParms>();
	const auto sopcache = dynamic_cast<SOP_VdbSolverCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);
	const GU_Detail* bgeo = cookparms.inputGeo(1);

	std::vector<openvdb::GridBase::ConstPtr> AGrid;
	openvdb::VectorGrid::ConstPtr BGrid;


	if (sopparms.getVeladvection()) {
		if (const UT_ErrorSeverity result = loadVDBs<openvdb::VectorGrid>(ageo, bgeo, AGrid, BGrid);
		    result != UT_ERROR_NONE) {
			cookparms.sopAddError(SOP_MESSAGE, "Failed to load VDB grids");
		}
	} else {
		if (const UT_ErrorSeverity result = loadVDBs<openvdb::FloatGrid>(ageo, bgeo, AGrid, BGrid);
		    result != UT_ERROR_NONE) {
			cookparms.sopAddError(SOP_MESSAGE, "Failed to load VDB grids");
		}
	}


	cudaStream_t stream;
	cudaStreamCreate(&stream);

	if (const UT_ErrorSeverity result = convertAndUpload<openvdb::VectorGrid>(sopcache->pBHandle, BGrid, &stream);
	    result != UT_ERROR_NONE) {
		cookparms.sopAddError(SOP_MESSAGE, "Failed to convert and upload velocity grid");
	}

	if (!sopparms.getVeladvection()) {
		std::vector<openvdb::FloatGrid::ConstPtr> floatGrids;
		for (const auto& grid : AGrid) {
			floatGrids.push_back(openvdb::gridConstPtrCast<openvdb::FloatGrid>(grid));
		}

		for (const auto& grid : floatGrids) {
			if (const UT_ErrorSeverity err =
			        processGrid<nanovdb::FloatGrid, openvdb::FloatGrid>(grid, sopcache, sopparms, detail, &stream);
			    err != UT_ERROR_NONE) {
				cookparms.sopAddError(SOP_MESSAGE, "Failed to process grid");
			}
		}

	} else {
		const auto castedInputGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>(AGrid[0]);
		if (const UT_ErrorSeverity err = processGrid<nanovdb::Vec3fGrid, openvdb::VectorGrid>(
		        castedInputGrid, sopcache, sopparms, detail, &stream);
		    err != UT_ERROR_NONE) {
			cookparms.sopAddError(SOP_MESSAGE, "Failed to process Velocity grid");
		}
	}


	printf("----------------------------------------------------\n");

	boss.end();
}


template <typename NanoVDBGridType, typename OpenVDBGridType>
UT_ErrorSeverity SOP_VdbSolverVerb::processGrid(const typename OpenVDBGridType::ConstPtr& grid,
                                                SOP_VdbSolverCache* sopcache, const SOP_VDBSolverParms& sopparms,
                                                GU_Detail* detail, const cudaStream_t* const stream) const {
	using Type = std::conditional_t<std::is_same_v<NanoVDBGridType, nanovdb::FloatGrid>, float, nanovdb::Vec3f>;
	if (std::is_same_v<OpenVDBGridType, openvdb::VectorGrid>) {
		sopcache->pAHandle = sopcache->pBHandle.copy<nanovdb::CudaDeviceBuffer>();
		sopcache->pAHandle.deviceUpload(*stream, false);
	} else {
		if (const UT_ErrorSeverity err = convertAndUpload<OpenVDBGridType>(sopcache->pAHandle, grid, stream);
		    err != UT_ERROR_NONE) {
			return err;
		}
	}

	NanoVDBGridType* gpuAGrid = sopcache->pAHandle.deviceGrid<Type>();
	const nanovdb::Vec3fGrid* gpuBGrid = sopcache->pBHandle.deviceGrid<nanovdb::Vec3f>();
	const NanoVDBGridType* cpuGrid = sopcache->pAHandle.grid<Type>();

	const uint32_t leafCount = cpuGrid->tree().nodeCount(0);
	const float voxelSize = cpuGrid->voxelSize()[0];
	const float deltaTime = sopparms.getTimestep();
	printf("leafCount: %d, voxelSize: %f, deltaTime: %f\n", leafCount, voxelSize, deltaTime);

	if constexpr (std::is_same_v<NanoVDBGridType, nanovdb::FloatGrid>) {
		auto temp = sopcache->pAHandle.copy<nanovdb::CudaDeviceBuffer>();
		temp.deviceUpload(*stream, false);
		nanovdb::FloatGrid* tempGrid = temp.deviceGrid<float>();
		thrust_kernel(tempGrid, gpuAGrid, gpuBGrid, leafCount, voxelSize, deltaTime, *stream);
	}

	if constexpr (std::is_same_v<NanoVDBGridType, nanovdb::Vec3fGrid>) {
		vel_thrust_kernel(gpuAGrid, gpuBGrid, leafCount, voxelSize, deltaTime, *stream);
	}

	sopcache->pAHandle.deviceDownload(*stream, true);

	if (const UT_ErrorSeverity err = convertToOpenVDBAndBuildGrid<OpenVDBGridType>(sopcache, detail, grid->getName());
	    err != UT_ERROR_NONE) {
		return err;
	}

	return UT_ERROR_NONE;
}


template <typename GridT>
UT_ErrorSeverity loadAGrid(const GU_Detail* aGeo, std::vector<openvdb::GridBase::ConstPtr>& AGrid) {
	ScopedTimer timer("Load input 1");


	const GA_PrimitiveGroup* group = aGeo->findPrimitiveGroup("dengroup");
	for (openvdb_houdini::VdbPrimCIterator it(aGeo, group); it; ++it) {
		if (auto grid = openvdb::gridConstPtrCast<GridT>((*it)->getConstGridPtr())) {
			AGrid.push_back(grid);
		}
	}

	if (AGrid.empty()) {
		return UT_ERROR_ABORT;
	}

	return UT_ERROR_NONE;
}


UT_ErrorSeverity loadBGrid(const GU_Detail* bGeo, openvdb::VectorGrid::ConstPtr& BGrid) {
	ScopedTimer timer("Load input 2");

	const GA_PrimitiveGroup* group = bGeo->findPrimitiveGroup("velgroup");
	for (openvdb_houdini::VdbPrimCIterator it(bGeo, group); it; ++it) {
		BGrid = openvdb::gridConstPtrCast<openvdb::VectorGrid>((*it)->getConstGridPtr());
		if (BGrid) break;
	}

	if (!BGrid) {
		return UT_ERROR_ABORT;
	}

	return UT_ERROR_NONE;
}


template <typename GridT>
UT_ErrorSeverity SOP_VdbSolverVerb::loadVDBs(const GU_Detail* aGeo, const GU_Detail* bGeo,
                                             std::vector<openvdb::GridBase::ConstPtr>& AGrid,
                                             openvdb::VectorGrid::ConstPtr& BGrid) {
	if (const UT_ErrorSeverity result = loadAGrid<GridT>(aGeo, AGrid); result != UT_ERROR_NONE) return result;

	return loadBGrid(bGeo, BGrid);
}


template <typename GridT>
UT_ErrorSeverity SOP_VdbSolverVerb::convertAndUpload(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& buffer,
                                                     const typename GridT::ConstPtr& grid,
                                                     const cudaStream_t* const stream) const {
	ScopedTimer timer("Convert to NanoVDB and Upload to GPU");

	if constexpr (std::is_same_v<GridT, openvdb::FloatGrid>) {
		buffer = nanovdb::createNanoGrid<openvdb::FloatGrid, float, nanovdb::CudaDeviceBuffer>(
		    *grid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable, 0);
	}

	if constexpr (std::is_same_v<GridT, openvdb::VectorGrid>) {
		buffer = nanovdb::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::CudaDeviceBuffer>(
		    *grid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable, 0);
	}

	if (!buffer.isEmpty()) {
		buffer.deviceUpload(*stream, false);
	} else {
		return UT_ERROR_ABORT;
	}

	return UT_ERROR_NONE;
}


template <typename GridT>
UT_ErrorSeverity SOP_VdbSolverVerb::convertToOpenVDBAndBuildGrid(SOP_VdbSolverCache* sopcache, GU_Detail* detail,
                                                                 const std::string& gridName) const {
	ScopedTimer conversionTimer("Convert to OpenVDB");
	const openvdb::GridBase::Ptr convertedGrid = nanovdb::nanoToOpenVDB(sopcache->pAHandle);
	const typename GridT::Ptr outputGrid = openvdb::gridPtrCast<GridT>(convertedGrid);

	ScopedTimer vdbCreationTimer("Create VDB grid");
	GU_PrimVDB::buildFromGrid(*detail, outputGrid, nullptr, gridName.c_str());

	return UT_ERROR_NONE;
}


const SOP_NodeVerb::Register<SOP_VdbSolverVerb> SOP_VdbSolverVerb::theVerb;
const SOP_NodeVerb* SOP_VdbSolver::cookVerb() const { return SOP_VdbSolverVerb::theVerb.get(); }