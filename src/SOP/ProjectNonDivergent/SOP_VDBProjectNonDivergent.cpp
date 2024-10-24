//
// Created by zphrfx on 03/10/2024.
//

#include "SOP_VDBProjectNonDivergent.hpp"
#include "MultigridSolver.hpp"

#include <UT/UT_DSOVersion.h>
#include <nanovdb/NanoVDB.h>


#include <Utils/Utils.hpp>
#include <Utils/ScopedTimer.hpp>
#include <Utils/OpenToNano.hpp>

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
	                                   SOP_HNanoVDBProjectNonDivergent::myConstructor,
	                                   SOP_HNanoVDBProjectNonDivergent::buildTemplates(), 1, 1, nullptr, 0));
}


const SOP_NodeVerb::Register<SOP_HNanoVDBProjectNonDivergentVerb> SOP_HNanoVDBProjectNonDivergentVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoVDBProjectNonDivergent::cookVerb() const {
	return SOP_HNanoVDBProjectNonDivergentVerb::theVerb.get();
}


void SOP_HNanoVDBProjectNonDivergentVerb::cook(const CookParms& cookparms) const {
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");
	auto sopcache = reinterpret_cast<SOP_HNanoVDBProjectNonDivergentCache*>(cookparms.cache());
	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* in_geo = cookparms.inputGeo(0);

	openvdb::VectorGrid::ConstPtr in_velocity = nullptr;
	for (openvdb_houdini::VdbPrimIterator it(in_geo); it; ++it) {
		if (const auto vdb = openvdb::gridPtrCast<openvdb::VectorGrid>((*it)->getGridPtr())) {
			in_velocity = vdb;
		}
	}
	if (in_velocity == nullptr) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	OpenVectorGrid open_out_data;
	{
		ScopedTimer timer("Extracting voxels from " + in_velocity->getName());
		extractFromOpenVDB<openvdb::VectorGrid, openvdb::Coord, openvdb::Vec3f>(in_velocity, open_out_data);
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	{
		ScopedTimer timer("Converting " + in_velocity->getName() + " to NanoVDB");
		pointToGridVectorToDevice(open_out_data, in_velocity->voxelSize()[0], sopcache->pHandle, stream);
	}

	NanoFloatGrid out_data;
	{
		ScopedTimer timer("Computing Divergence");

		out_data.size = open_out_data.size;
		out_data.pCoords = new nanovdb::Coord[out_data.size];
		memcpy_s(out_data.pCoords, out_data.size * sizeof(nanovdb::Coord), (nanovdb::Coord*)open_out_data.pCoords, open_out_data.size * sizeof(openvdb::Coord));

		out_data.pValues = new float[out_data.size];

		ComputeDivergence(sopcache->pHandle, out_data, stream);
	}


	{
		ScopedTimer timer("Building Divergence Grid");

		const openvdb::FloatGrid::Ptr out = openvdb::FloatGrid::create();
		out->setGridClass(openvdb::GRID_FOG_VOLUME);
		out->setTransform(openvdb::math::Transform::createLinearTransform(in_velocity->voxelSize()[0]));

		detail->clearAndDestroy();

		openvdb::tree::ValueAccessor<openvdb::FloatTree> accessor(out->tree());

		for (size_t i = 0; i < out_data.size; ++i) {
			auto& coord = out_data.pCoords[i];
			auto value = out_data.pValues[i];
			accessor.setValue(openvdb::Coord(coord.x(), coord.y(), coord.z()), value);
		}

		GU_PrimVDB::buildFromGrid(*detail, out, nullptr, "divergence");
	}
}
