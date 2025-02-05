//
// Created by zphrfx on 03/10/2024.
//
#include "SOP_VDBProjectNonDivergent.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>
#include <PRM/PRM_TemplateBuilder.h>

#include "Utils/GridBuilder.hpp"
#include "Utils/ScopedTimer.hpp"

#define NANOVDB_USE_OPENVDB
#include <nanovdb/tools/CreateNanoGrid.h>


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
	parm {
		name "iterations"
		label "Iterations"
        type    integer
        size    1
        range   { 1! 100 }
	}
    parm {
        name    "outdiv"
        label   "Output Divergence"
        type    toggle
        default { "0" }
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
	                                   SOP_HNanoVDBProjectNonDivergent::myConstructor, SOP_HNanoVDBProjectNonDivergent::buildTemplates(), 1,
	                                   1, nullptr, OP_FLAG_GENERATOR));
}


const SOP_NodeVerb::Register<SOP_HNanoVDBProjectNonDivergentVerb> SOP_HNanoVDBProjectNonDivergentVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoVDBProjectNonDivergent::cookVerb() const { return SOP_HNanoVDBProjectNonDivergentVerb::theVerb.get(); }


void SOP_HNanoVDBProjectNonDivergentVerb::cook(const CookParms& cookparms) const {
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");
	auto sopcache = reinterpret_cast<SOP_HNanoVDBProjectNonDivergentCache*>(cookparms.cache());
	const auto& sopparms = cookparms.parms<SOP_VDBProjectNonDivergentParms>();

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* in_geo = cookparms.inputGeo(0);

	openvdb::VectorGrid::Ptr in_velocity = nullptr;
	if (auto err = loadGrid<openvdb::VectorGrid>(in_geo, in_velocity, sopparms.getVelgrid()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}


	cudaStream_t stream;
	cudaStreamCreate(&stream);

	using SrcGridT = openvdb::FloatGrid;
	using DstBuildT = nanovdb::ValueOnIndex;
	using BufferT = nanovdb::cuda::DeviceBuffer;


	SrcGridT::Ptr domain = openvdb::createGrid<openvdb::FloatGrid>();
	{
		ScopedTimer timer("Merging topology");
		domain->topologyUnion(*in_velocity);
	}

	HNS::GridIndexedData data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &data);
	{
		ScopedTimer t("Extracting data from OpenVDB");

		builder.addGrid(in_velocity, "vel");
		builder.build();
	}

	{
		ScopedTimer timer("Copying data to device");
		nanovdb::GridHandle<BufferT> handle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*domain, 1u, false, false, 0);
		handle.deviceUpload(stream, true);
		const auto* gpuGrid = handle.deviceGrid<DstBuildT>();

		Divergence_idx(gpuGrid, data, sopparms.getIterations(), in_velocity->voxelSize()[0], stream);
	}

	{
		ScopedTimer timer("Writing grids");

		openvdb::VectorGrid::Ptr div = builder.writeIndexGrid<openvdb::VectorGrid>("vel", in_velocity->voxelSize()[0]);

		GU_PrimVDB::buildFromGrid(*detail, div, nullptr, div->getName().c_str());
	}

	cudaStreamDestroy(stream);
}
