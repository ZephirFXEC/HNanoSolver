//
// Created by zphrfx on 29/08/2024.
//

#include "SOP_VDBFromGrid.hpp"

#include <GA/GA_SplittableRange.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include "Utils/OpenToNano.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

extern "C" void pointToGrid(const std::vector<nanovdb::Coord>& coords, const std::vector<float>& values, float voxelSize, NanoFloatGrid& out_data);


// Assuming openvdb::Coord and nanovdb::Coord have compatible constructors
std::vector<nanovdb::Coord> convertCoordVector(const std::vector<openvdb::Coord>& openVDBCoords) {
	std::vector<nanovdb::Coord> nanoCoords;
	nanoCoords.reserve(openVDBCoords.size());  // Preallocate for efficiency

	std::transform(openVDBCoords.begin(), openVDBCoords.end(), std::back_inserter(nanoCoords),
		[](const openvdb::Coord& coord) {
			return nanovdb::Coord(coord.x(), coord.y(), coord.z());
		});

	return nanoCoords;
}


const char* const SOP_HNanoVDBFromGridVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
        name        "attribs"
        label       "Attribute to rasterize"
        type        string
        default     { "density" }
        parmtag     { "sop_input" "0" }
    }
    parm {
        name    "div"
        label   "Divisions"
        type    integer
        default { "1" }
        range   { 1! 12 }
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

PRM_Template* SOP_HNanoVDBFromGrid::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBFromGrid.cpp", SOP_HNanoVDBFromGridVerb::theDsFile);
	if (templ.justBuilt()) {
		templ.setChoiceListPtr("attribs", &SOP_Node::pointAttribMenu);
	}
	return templ.templates();
}

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanofromgrid", "HNanoFromGrid", SOP_HNanoVDBFromGrid::myConstructor,
	                                   SOP_HNanoVDBFromGrid::buildTemplates(), 1, 1, nullptr, 0));
}


const SOP_NodeVerb::Register<SOP_HNanoVDBFromGridVerb> SOP_HNanoVDBFromGridVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoVDBFromGrid::cookVerb() const { return SOP_HNanoVDBFromGridVerb::theVerb.get(); }


void SOP_HNanoVDBFromGridVerb::cook(const CookParms& cookparms) const {
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");
	const auto& sopparms = cookparms.parms<SOP_VDBFromGridParms>();
	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* in_geo = cookparms.inputGeo(0);

	openvdb::FloatGrid::ConstPtr grid = nullptr;
	for (openvdb_houdini::VdbPrimIterator it(in_geo); it; ++it) {
		if (const auto vdb = openvdb::gridPtrCast<openvdb::FloatGrid>((*it)->getGridPtr())) {
			grid = vdb;
		}
	}

	if (grid == nullptr) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	std::vector<openvdb::Coord> pos;
	std::vector<float> value;
	{
		ScopedTimer timer("Extracting points from input geometry");
		extractFromOpenVDB(grid, pos, value);
	}

	NanoFloatGrid out_data;
	{
		ScopedTimer timer("Creating NanoVDB grid");
		const auto nanoPos = convertCoordVector(pos);
		pointToGrid(nanoPos, value, sopparms.getVoxelsize(), out_data);
	}


}

