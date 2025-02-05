//
// Created by zphrfx on 29/08/2024.
//


#define NANOVDB_USE_OPENVDB

#include "SOP_VDBFromGrid.hpp"

#include <UT/UT_DSOVersion.h>
#include <cuda_runtime_api.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include "Utils/GridBuilder.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"


extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid, HNS::GridIndexedData& data, float dt, float voxelSize, cudaStream_t stream );

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
	                                   SOP_HNanoVDBFromGrid::buildTemplates(), 2, 2, nullptr, OP_FLAG_GENERATOR));
}


const SOP_NodeVerb::Register<SOP_HNanoVDBFromGridVerb> SOP_HNanoVDBFromGridVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoVDBFromGrid::cookVerb() const { return SOP_HNanoVDBFromGridVerb::theVerb.get(); }


void SOP_HNanoVDBFromGridVerb::cook(const CookParms& cookparms) const {
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");
	const auto& sopparms = cookparms.parms<SOP_VDBFromGridParms>();
	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* in1_geo = cookparms.inputGeo(0);
	const GU_Detail* in2_geo = cookparms.inputGeo(1);

	std::vector<openvdb::FloatGrid::Ptr> AGrid;
	if (auto err = loadGrid<openvdb::FloatGrid>(in1_geo, AGrid, sopparms.getAgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load grids");
	}

	std::vector<openvdb::VectorGrid::Ptr> BGrid;
	if (auto err = loadGrid<openvdb::VectorGrid>(in2_geo, BGrid, sopparms.getBgroup()); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load grids");
	}


	using SrcGridT = openvdb::FloatGrid;
	using DstBuildT = nanovdb::ValueOnIndex;
	using BufferT = nanovdb::cuda::DeviceBuffer;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// merge topolgy

	openvdb::FloatGrid::Ptr domain = openvdb::createGrid<openvdb::FloatGrid>();
	{
		ScopedTimer t("Merging topology");
		domain->topologyUnion(*AGrid[0]);
		domain->topologyUnion(*BGrid[0]);
	}

	HNS::GridIndexedData data;
	HNS::IndexGridBuilder<SrcGridT> builder(domain, &data);
	{
		ScopedTimer t("Extracting data from OpenVDB");

		builder.addGrid(AGrid[0], "density");
		builder.addGrid(BGrid[0], "velocity");
		builder.build();
	}

	{
		nanovdb::GridHandle<BufferT> idxHandle =
		    nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*domain, 1u, false, false, 0);

		idxHandle.deviceUpload(stream, false);
		const auto* gpuGrid = idxHandle.deviceGrid<DstBuildT>();

		ScopedTimer timer("Launching kernels");
		const float deltaTime = static_cast<float>(sopparms.getTimestep());
		launch_kernels(gpuGrid, data, deltaTime, AGrid[0]->voxelSize()[0], stream);
	}


	{
		ScopedTimer timer("Building Grids");


		tbb::parallel_invoke(
		    [&] {
			    const auto density = builder.writeIndexGrid<openvdb::FloatGrid>("density", AGrid[0]->voxelSize()[0]);
			    GU_PrimVDB::buildFromGrid(*detail, density, nullptr, density->getName().c_str());
		    },
		    [&] {
			    const auto velocity = builder.writeIndexGrid<openvdb::VectorGrid>("velocity", AGrid[0]->voxelSize()[0]);
			    GU_PrimVDB::buildFromGrid(*detail, velocity, nullptr, velocity->getName().c_str());
		    });
	}
}
