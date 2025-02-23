//
// Created by zphrfx on 29/08/2024.
//


#define NANOVDB_USE_OPENVDB

#include "SOP_VDBFromGrid.hpp"

#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include "Cuda/BrickMap/BrickMap.cuh"
#include "Utils/GridBuilder.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

extern "C" void launch_kernels(HNS::GridIndexedData& data, float dt, float voxelSize, cudaStream_t stream);
extern "C" void gpu_index_grid(HNS::GridIndexedData& data, float voxelSize, const cudaStream_t& stream);
extern "C" void accessBrick(const BrickMap& brickMap);
extern "C" void InitVel(const BrickMap& brickMap);
extern "C" void advect(const BrickMap& brickMap, float dt);

const char* const SOP_HNanoVDBFromGridVerb::theDsFile = R"THEDSFILE(
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

PRM_Template* SOP_HNanoVDBFromGrid::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBFromGrid.cpp", SOP_HNanoVDBFromGridVerb::theDsFile);
	if (templ.justBuilt()) {
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
		templ.setChoiceListPtr("bgroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanofromgrid", "HNanoFromGrid", SOP_HNanoVDBFromGrid::myConstructor,
	                                   SOP_HNanoVDBFromGrid::buildTemplates(), 0, 0, nullptr, OP_FLAG_GENERATOR));
}


const SOP_NodeVerb::Register<SOP_HNanoVDBFromGridVerb> SOP_HNanoVDBFromGridVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoVDBFromGrid::cookVerb() const { return SOP_HNanoVDBFromGridVerb::theVerb.get(); }


void SOP_HNanoVDBFromGridVerb::cook(const CookParms& cookparms) const {
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");
	const auto& sopparms = cookparms.parms<SOP_VDBFromGridParms>();

	auto sopcache = reinterpret_cast<SOP_HNanoVDBFromGridCache*>(cookparms.cache());
	if (!sopcache) {
		sopcache = new SOP_HNanoVDBFromGridCache();
	}

	GU_Detail* detail = cookparms.gdh().gdpNC();
	OP_Context context = cookparms.getContext();
	const int currentFrame = context.getFrame();

	BrickMap& brickMap = *sopcache->brickMapSingleton.getBrickMap();

	if (currentFrame == 1) {
		if (!brickMap.allocateBrickAt(BrickCoord(0, 0, 0))) {
			printf("Failed to allocate brick at (0, 0, 0)\n");
			return;
		}
		accessBrick(brickMap);

		if (!brickMap.allocateBrickAt(BrickCoord(1, 0, 0))) {
			printf("Failed to allocate brick at (1, 0, 0)\n");
			return;
		}

		InitVel(brickMap);

	}


	// Always perform advection step
	{
		ScopedTimer timer("BrickMap::Advection");
		advect(brickMap, sopparms.getTimestep());
	}

	std::vector<BrickCoord> brickCoord;
	{
		ScopedTimer timer("BrickMap::getActiveBricks");
		brickCoord = brickMap.getActiveBricks();
		printf("Active bricks: %llu\n", brickCoord.size());
	}


	// Assume:
	//   'detail' is a GU_Detail*
	//   'brickCoord' is a std::vector<BrickCoord>
	//   'sopcache->brickMap.getBrickAtHost(coord)' returns a pointer to 32^3 voxels.
	//   Each 'Voxel' struct contains at least a 'float density' field.

	for (size_t b = 0; b < brickCoord.size(); ++b) {
		const BrickCoord& coord = brickCoord[b];

		// Create a brand-new FloatGrid for this brick
		openvdb::FloatGrid::Ptr out = openvdb::FloatGrid::create(/*background*/ 0.0f);
		// Give each grid a unique name, e.g. "density_0", "density_1", ...
		out->setName("density_" + std::to_string(b));

		// Get the accessor for writing voxel values
		auto acc = out->getAccessor();

		// Retrieve the brick (32^3 voxels) from the BrickMap
		const Voxel* brick = brickMap.getBrickAtHost(coord);
		if (!brick) continue;  // Safety check if it might be null

		// Fill this grid with the brick’s density data
		for (int i = 0; i < 32 * 32 * 32; i++) {
			// Convert linear index i -> local (x,y,z) in [0..31]
			openvdb::Coord local(i % 32, (i / 32) % 32, i / (32 * 32));

			// Compute a world offset if each brick is placed at "coord * 32"
			openvdb::Coord world(local.x() + int(coord[0] * 32), local.y() + int(coord[1] * 32), local.z() + int(coord[2] * 32));

			// Write the density
			acc.setValue(world, brick[i].density);
		}

		// Build a GU_PrimVDB (Volume primitive) for display from this grid
		GU_PrimVDB::buildFromGrid(*detail, out, /*transform*/ nullptr, out->getName().c_str());
	}
}
