#include "SOP_VDBAdvectVelocity.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_PrimVolume.h>
#include <UT/UT_DSOVersion.h>
#include <tbb/enumerable_thread_specific.h>

#include "Utils/GridBuilder.hpp"
#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

#define NANOVDB_USE_OPENVDB
#include <nanovdb/tools/CreateNanoGrid.h>




void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanoadvectvelocity", "HNanoAdvectVelocity", SOP_HNanoAdvectVelocity::myConstructor,
	                                   SOP_HNanoAdvectVelocity::buildTemplates(), 1, 1, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_HNanoAdvectVelocityVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
		name	"agroup"
		label	"Velocity Volumes Advected"
		type	string
		default	{ "" }
		parmtag	{ "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Primitives,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
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


PRM_Template* SOP_HNanoAdvectVelocity::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBAdvectedVelocity.cpp", SOP_HNanoAdvectVelocityVerb::theDsFile);
	if (templ.justBuilt()) {
		templ.setChoiceListPtr("agroup", &SOP_Node::namedVolumesMenu);
	}
	return templ.templates();
}


void SOP_HNanoAdvectVelocityVerb::cook(const CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_VDBAdvectVelocityParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoAdvectVelocityCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* ageo = cookparms.inputGeo(0);

	openvdb::VectorGrid::Ptr AGrid = nullptr;

	if (auto err = loadGrid<openvdb::VectorGrid>(ageo, AGrid, sopparms.getAgroup()); err != UT_ERROR_NONE) {
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
		domain->topologyUnion(*AGrid);
	}

	HNS::GridIndexedData data;
	HNS::IndexGridBuilder<openvdb::FloatGrid> builder(domain, &data);
	{
		ScopedTimer t("Extracting data from OpenVDB");

		builder.addGrid(AGrid, "vel");
		builder.build();
	}

	{
		nanovdb::GridHandle<BufferT> idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*domain, 1u, false, false, 0);

		idxHandle.deviceUpload(stream, false);
		const auto* gpuGrid = idxHandle.deviceGrid<DstBuildT>();

		ScopedTimer timer_kernel("Launching kernels");
		const float deltaTime = static_cast<float>(sopparms.getTimestep());
		AdvectIndexGridVelocity(gpuGrid, data, deltaTime, AGrid->voxelSize()[0], stream);
	}

	{
		ScopedTimer timer("Writing grids");


		openvdb::VectorGrid::Ptr vel = builder.writeIndexGrid<openvdb::VectorGrid>("vel", AGrid->voxelSize()[0]);

		GU_PrimVDB::buildFromGrid(*detail, vel, nullptr, vel->getName().c_str());
	}

	cudaStreamDestroy(stream);
}


const SOP_NodeVerb::Register<SOP_HNanoAdvectVelocityVerb> SOP_HNanoAdvectVelocityVerb::theVerb;
const SOP_NodeVerb* SOP_HNanoAdvectVelocity::cookVerb() const { return SOP_HNanoAdvectVelocityVerb::theVerb.get(); }