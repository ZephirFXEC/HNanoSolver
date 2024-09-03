//
// Created by zphrfx on 29/08/2024.
//

#include "SOP_ReadWriteTest.hpp"

#include <UT/UT_DSOVersion.h>
#include <GA/GA_SplittableRange.h>
#include <vector>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include "Utils/ScopedTimer.hpp"
#include "Utils/Utils.hpp"

extern "C" void pointToGrid(const Grid& gridData, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&);

const char* const SOP_ReadWriteTestVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
	parm {
		name "voxelsize"
		label "Voxel Size"
        type    float
        size    1
        default { "0.5" }
	}
}
)THEDSFILE";

PRM_Template* SOP_ReadWriteTest::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VDBSolver.cpp", SOP_ReadWriteTestVerb::theDsFile);
	return templ.templates();
}

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hreadwrite", "HReadWrite", SOP_ReadWriteTest::myConstructor,
	                                   SOP_ReadWriteTest::buildTemplates(), 1, 1, nullptr, OP_FLAG_GENERATOR));
}


const SOP_NodeVerb::Register<SOP_ReadWriteTestVerb> SOP_ReadWriteTestVerb::theVerb;

const SOP_NodeVerb* SOP_ReadWriteTest::cookVerb() const { return SOP_ReadWriteTestVerb::theVerb.get(); }


void SOP_ReadWriteTestVerb::cook(const CookParms& cookparms) const {
	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");
	const auto& sopparms = cookparms.parms<SOP_ReadWriteTestParms>();
	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GEO_Detail* const in_geo = cookparms.inputGeo(0);

	if (!in_geo) {
		cookparms.sopAddError(SOP_MESSAGE, "No input geometry found");
	}


	const GA_ROHandleF attrib(detail->findFloatTuple(GA_ATTRIB_POINT, "density"));
	if(!attrib.isValid()) {
		cookparms.sopAddError(SOP_MESSAGE, "No density attribute found");
	}

	std::vector<nanovdb::Coord> coords;
	std::vector<float> values;


	{ // TODO: idk what I did there but without mutex it crashes
		ScopedTimer timer("Extracting points");
		std::mutex mutex;
		UTparallelFor(GA_SplittableRange(in_geo->getPointRange()), [&](const GA_Range& range) {
			// Thread-local storage for each thread
			std::vector<nanovdb::Coord> local_coords;
			std::vector<float> local_values;

			for(GA_Iterator it(range); !it.atEnd(); ++it)
			{
				UT_Vector3F pos = in_geo->getPos3(it.getOffset());
				float value = attrib.get(it.getOffset());
				local_coords.emplace_back(pos[0], pos[1], pos[2]);
				local_values.push_back(value);
			}

			// Lock and append local results to shared vectors
			{
				std::lock_guard<std::mutex> lock(mutex);
				coords.insert(coords.end(), local_coords.begin(), local_coords.end());
				values.insert(values.end(), local_values.begin(), local_values.end());
			}
		});
	}

	const Grid data = {coords, values, static_cast<float>(sopparms.getVoxelsize())};

	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle;
	{
		const auto name = "Create NanoVDB Grid from points";
		boss.start(name);
		ScopedTimer timer(name);
		pointToGrid(data, handle);
		boss.end();
	}

	if(handle.isEmpty()) {
		cookparms.sopAddError(SOP_MESSAGE, "Failed to create grid");
	}

	handle.deviceDownload();
	{
		const auto name = "Create VDB Grid from NanoVDB Grid";
		boss.start(name);
		ScopedTimer timer(name);
		const auto grid = nanovdb::nanoToOpenVDB(handle);
		const openvdb::FloatGrid::Ptr vdbGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
		GU_PrimVDB::buildFromGrid(*detail, vdbGrid, nullptr, "density");
		boss.end();
	}

}

void LoadVDBs(const SOP_NodeVerb::CookParms& cookparms, openvdb_houdini::HoudiniInterrupter& boss,
              const GEO_Detail* const in_geo) {
	std::vector<GU_PrimVDB*> prims;

	for (openvdb_houdini::VdbPrimIterator it(in_geo); it; ++it) {
		if (boss.wasInterrupted()) {
			throw std::runtime_error("processing was interrupted");
		}
		prims.push_back(it.getPrimitive());
	}

	if (prims.empty()) {
		cookparms.sopAddError(SOP_MESSAGE, "First input must contain VDBs!");
	}

	{
		ScopedTimer timer("Making grids unique");
		for (const auto& prim : prims) {
			prim->makeGridUnique();
		}
	}

	std::vector<openvdb::GridBase::Ptr> grids;
	{
		ScopedTimer timer("Extracting grids");
		for (const auto prim : prims) {
			grids.push_back(prim->getGridPtr());
		}
	}

	auto printOp = [](const openvdb::GridBase& in) { in.print(std::cout, 1); };
	for (const auto& grid : grids) {
		grid->apply<openvdb_houdini::VolumeGridTypes>(printOp);
	}
}