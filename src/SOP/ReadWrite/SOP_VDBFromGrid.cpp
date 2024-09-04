//
// Created by zphrfx on 29/08/2024.
//

#include "SOP_VDBFromGrid.hpp"

#include <GA/GA_SplittableRange.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include <vector>

#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

#include <GEO/GEO_AttributeHandle.h>

extern "C" void pointToGrid(const Grid& gridData, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&);

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
	if (templ.justBuilt())
	{
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
	const GEO_Detail* const in_geo = cookparms.inputGeo(0);

	// Channel manager has time info for us
	const CH_Manager *chman = OPgetDirector()->getChannelManager();
	// This is the frame that we're cooking at...
	fpreal currframe = chman->getSample(cookparms.getCookTime());

	if (!in_geo) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}

	const GA_ROHandleF attrib(detail->findFloatTuple(GA_ATTRIB_POINT, "density"));
	if (!attrib.isValid()) {
		cookparms.sopAddMessage(SOP_MESSAGE, "No attribute found to rasterize");
	}

	std::vector<openvdb::Coord> coords;
	std::vector<float> values;


	{
		ScopedTimer timer("Extracting points");
		GA_Offset block_start, block_end;
		for (GA_Iterator pageI(detail->getPointRange()); pageI.blockAdvance(block_start, block_end); ) {
			for (GA_Offset ptoff = block_start; ptoff < block_end; ++ptoff)  {
				UT_Vector3F pos = in_geo->getPos3(ptoff);
				float value = attrib.get(ptoff);
				coords.emplace_back(pos[0], pos[1], pos[2]);
				values.push_back(value);
			}
		}
	}

	/*
	 * TODO: Idea: create a kernel that loops over the nanovdb grid and output [pos, values]
	 * so i can build the openvdb grid in parallel
	 */

	const size_t numCores = sopparms.getDiv();
	const auto chunkSize = coords.size() / numCores;

	std::mutex detailMutex;  // Mutex to protect shared resources

	auto processChunk = [&](const size_t start, const size_t end) {
		ScopedTimer timer("Creating VDB grids");

		Grid chunkData = {
			std::vector<openvdb::Coord>(coords.begin() + start, coords.begin() + end),
			std::vector<float>(values.begin() + start, values.begin() + end),
			static_cast<float>(sopparms.getVoxelsize())
		};


		const openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
		grid->setGridClass(openvdb::GRID_FOG_VOLUME);
		grid->setTransform(openvdb::math::Transform::createLinearTransform(sopparms.getVoxelsize()));
		grid->setName("density");

		auto accessor = grid->getAccessor();

		for (size_t j = start; j < end; ++j) {
			auto& coord = coords[j];
			auto& value = values[j];
			accessor.setValue(openvdb::Coord(coord.x(), coord.y(), coord.z()), value);
		}

		{
			std::lock_guard<std::mutex> lock(detailMutex);
			GU_PrimVDB::buildFromGrid(*detail, grid);
		}
	};

	detail->clearAndDestroy();
	openvdb::initialize();

	UTparallelFor(UT_BlockedRange<size_t>(0, numCores), [&](const UT_BlockedRange<size_t>& range) {
		for (size_t i = range.begin(); i < range.end(); ++i) {
			const size_t start = i * chunkSize;
			const size_t end = (i == numCores - 1) ? coords.size() : (i + 1) * chunkSize;
			processChunk(start, end);
		}
	});
}