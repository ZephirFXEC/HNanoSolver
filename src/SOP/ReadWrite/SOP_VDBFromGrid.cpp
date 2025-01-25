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

#define NANOVDB_USE_OPENVDB

#include "SOP_VDBFromGrid.hpp"

#include <UT/UT_DSOVersion.h>
#include <cuda_runtime_api.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include "Utils/OpenToNano.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"



extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>*, HNS::IndexFloatGrid& , cudaStream_t stream);

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
		templ.setChoiceListPtr("bgroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanofromgrid", "HNanoFromGrid", SOP_HNanoVDBFromGrid::myConstructor,
	                                   SOP_HNanoVDBFromGrid::buildTemplates(), 2, 2, nullptr, 0));
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



	using SrcGridT  = openvdb::FloatGrid;
	using DstBuildT = nanovdb::ValueOnIndex;
	using BufferT   = nanovdb::cuda::DeviceBuffer;

	cudaStream_t stream;
	cudaStreamCreate(&stream);


	HNS::IndexFloatGrid toSample;
	{
		ScopedTimer t("Extracting data from OpenVDB");
		HNS::extractToGlobalIdx(BGrid[0], AGrid[0], toSample);
	}

	{
		ScopedTimer timer("NanoVDB conversion");
		nanovdb::GridHandle<BufferT> idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*AGrid[0], 1u, false , false, 0);

		// auto idxHandle = nanovdb::tools::openToNanoVDB<SrcGridT, DstBuildT, BufferT>(*AGrid[0], 1u, false, false, 0);
		idxHandle.deviceUpload(stream, false);
		const auto* gpuGrid = idxHandle.deviceGrid<DstBuildT>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

		launch_kernels(gpuGrid, toSample, stream); // Call a host method to print a grid value on both the CPU and GPU

		cudaStreamDestroy(stream); // Destroy the CUDA stream
	}

	/*
	{
		ScopedTimer timer("OpenVDB To NanoVDB Index Cuda");

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		openToNanoIndex(AGrid[0], stream);

		cudaStreamDestroy(stream); // Destroy the CUDA stream
	}
	*/
}
