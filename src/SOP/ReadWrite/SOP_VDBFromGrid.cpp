//
// Created by zphrfx on 29/08/2024.
//

/* TODO: extract data from openvdb containing :
- Values
- Coords
- Value[Coord] mapping

then no conversion to nanovdb is needed.
custom sampler as to be written.
Run the kernels on the GPU

export back value / coord to the CPU to build the grid.
*/


#include "SOP_VDBFromGrid.hpp"

#include <UT/UT_DSOVersion.h>
#include <UT/UT_Tracing.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "Utils/OpenToNano.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

extern "C" void pointToGridFloat(const HNS::OpenFloatGrid& in_data, float voxelSize, HNS::NanoFloatGrid& out_data,
                                 const cudaStream_t& stream);


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
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
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

	std::vector<openvdb::FloatGrid::Ptr> AGrid;
	if (auto err = loadGrid<openvdb::FloatGrid>(in_geo, AGrid, sopparms.getAgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load grids");
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	HNS::OpenFloatGrid open_out_data;
	{
		ScopedTimer timer("Extracting voxels from " + AGrid[0]->getName());
		HNS::extractFromOpenVDB<openvdb::FloatGrid, float>(AGrid[0], open_out_data);
	}

	HNS::NanoFloatGrid out_data;
	{
		ScopedTimer timer("Creating " + AGrid[0]->getName() + " NanoVDB grid");

		pointToGridFloat(open_out_data, sopparms.getVoxelsize(), out_data, stream);
	}

	detail->clearAndDestroy();

	{
		ScopedTimer timer("Building " + AGrid[0]->getName() + " grid");

		openvdb::FloatGrid::Ptr out = openvdb::FloatGrid::create();
		out->setGridClass(openvdb::GRID_FOG_VOLUME);
		out->setTransform(openvdb::math::Transform::createLinearTransform(sopparms.getVoxelsize()));
		out->setName(AGrid[0]->getName());

		openvdb::tree::ValueAccessor<openvdb::FloatTree> valueAccessor(out->tree());

		for (size_t i = 0; i < out_data.size; ++i) {
			const auto& coord = out_data.pCoords()[i];
			const auto& value = out_data.pValues()[i];
			valueAccessor.setValueOn(openvdb::Coord(coord.x(), coord.y(), coord.z()), value);
		}

		GU_PrimVDB::buildFromGrid(*detail, out, nullptr, out->getName().c_str());
	}

	cudaStreamDestroy(stream);
}
