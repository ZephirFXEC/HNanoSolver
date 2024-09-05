#include "SOP_VDBAdvect.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"


void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvect", "HNanoAdvect", SOP_HNanoVDBAdvect::myConstructor,
	                                   SOP_HNanoVDBAdvect::buildTemplates(), 2, 2, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_HNanoVDBAdvectVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
		name	"agroup"
		label	"Density Volumes"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
		parmtag	{ "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
		parmtag	{ "script_action_icon" "BUTTONS_reselect" }
    }
    parm {
		name	"bgroup"
		label	"Velocity Volume"
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


PRM_Template* SOP_HNanoVDBAdvect::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBAdvect.cpp", SOP_HNanoVDBAdvectVerb::theDsFile);
	if (templ.justBuilt()) {
		// They don't work, for now all the FloatGrid found in the 1st input will be advected
		// and the velocity field will be the first found in the 2nd input.
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
		templ.setChoiceListPtr("bgroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}


void SOP_HNanoVDBAdvectVerb::cook(const SOP_NodeVerb::CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_VDBAdvectParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoVDBAdvectCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);
	const GU_Detail* bgeo = cookparms.inputGeo(1);

	std::vector<openvdb::FloatGrid::Ptr> AGrid;
	std::vector<openvdb::VectorGrid::Ptr> BGrid;  // Velocity grid ( len = 1 )

	if (auto err = loadGrid<openvdb::FloatGrid>(ageo, AGrid, sopparms.getAgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load density grid");
	}

	if (auto err = loadGrid<openvdb::VectorGrid>(bgeo, BGrid, sopparms.getBgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load velocity grid");
	}


	cudaStream_t stream;
	cudaStreamCreate(&stream);

	{
		boss.start();
		ScopedTimer timer("Total Advection");

		{
			boss.start();
			ScopedTimer timer("Creating NanoVDB Velocity grids");

			sopcache->pBHandle =
			    nanovdb::createNanoGrid<openvdb::VectorGrid, nanovdb::Vec3f, nanovdb::CudaDeviceBuffer>(
			        *BGrid[0], nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable, 0);
			sopcache->pBHandle.deviceUpload(stream, false);

			boss.end();
		}

		for (auto& grid : AGrid) {
			nanovdb::Coord* h_coords = nullptr;
			float* h_values = nullptr;
			size_t count = 0;

			{
				ScopedTimer timer("Converting " + grid->getName() + " to NanoVDB");

				sopcache->pAHandle = nanovdb::createNanoGrid<openvdb::FloatGrid, float, nanovdb::CudaDeviceBuffer>(
				    *grid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable, 0);
				sopcache->pAHandle.deviceUpload(stream, false);
			}


			float voxelSize = 0.5f;
			{
				ScopedTimer timer("Computing " + grid->getName() + " advection");

				nanovdb::FloatGrid* gpuAGrid = sopcache->pAHandle.deviceGrid<float>();
				const nanovdb::Vec3fGrid* gpuBGrid = sopcache->pBHandle.deviceGrid<nanovdb::Vec3f>();
				const nanovdb::FloatGrid* cpuGrid = sopcache->pAHandle.grid<float>();

				const uint32_t leafCount = cpuGrid->tree().nodeCount(0);
				voxelSize = static_cast<float>(cpuGrid->voxelSize()[0]);
				const auto deltaTime = static_cast<float>(sopparms.getTimestep());

				h_coords = new nanovdb::Coord[512 * leafCount];
				h_values = new float[512 * leafCount];

				thrust_kernel(gpuAGrid, gpuBGrid, leafCount, voxelSize, deltaTime, stream, h_coords, h_values, count);
			}

			{
				ScopedTimer timer("Building Grid " + grid->getName());

				const openvdb::FloatGrid::Ptr out = openvdb::FloatGrid::create();
				out->setGridClass(openvdb::GRID_FOG_VOLUME);
				out->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
				out->setName(grid->getName());

				auto accessor = grid->getAccessor();

				for (size_t i = 0; i < count; ++i) {
					auto& coord = h_coords[i];
					auto& value = h_values[i];
					accessor.setValue(openvdb::Coord(coord.x(), coord.y(), coord.z()), value);
				}

				GU_PrimVDB::buildFromGrid(*detail, grid, nullptr, out->getName().c_str());
			}

			delete[] h_coords;
			delete[] h_values;
		}

		boss.end();
	}

	cudaStreamDestroy(stream);
}


template <typename GridT>
UT_ErrorSeverity SOP_HNanoVDBAdvectVerb::loadGrid(const GU_Detail* aGeo, std::vector<typename GridT::Ptr>& grid,
                                                  const UT_StringHolder& group) const {
	const GA_PrimitiveGroup* groupRef = aGeo->findPrimitiveGroup(group);
	for (openvdb_houdini::VdbPrimIterator it(aGeo, groupRef); it; ++it) {
		if (auto vdb = openvdb::gridPtrCast<GridT>((*it)->getGridPtr())) {
			grid.push_back(vdb);
		}
	}

	if (grid.empty()) {
		return UT_ERROR_ABORT;
	}

	return UT_ERROR_NONE;
}


const SOP_NodeVerb::Register<SOP_HNanoVDBAdvectVerb> SOP_HNanoVDBAdvectVerb::theVerb;
const SOP_NodeVerb* SOP_HNanoVDBAdvect::cookVerb() const { return SOP_HNanoVDBAdvectVerb::theVerb.get(); }