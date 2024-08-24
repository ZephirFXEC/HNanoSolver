#ifndef __SOP_VDBSOLVER_HPP__
#define __SOP_VDBSOLVER_HPP__


#include "OpenVDB_Utils/SOP_NodeVDB.hpp"
#include "OpenVDB_Utils/Utils.hpp"
#include "Utils/IncomingDataCache.hpp"


using namespace openvdb_houdini;

namespace VdbSolver {
class SOP_VdbSolver final : public SOP_NodeVDB {
   public:
	// node contructor for HDK
	static OP_Node* myConstructor(OP_Network*, const char*, OP_Operator*);

	class Cache final : public SOP_VDBCacheOptions {
	   public:
		OP_ERROR cookVDBSop(OP_Context&) override;

		template <typename GridType>
		typename GridType::ConstPtr processGrid(const GridCPtr& in);


		// GridType is either openvdb::FloatGrid or openvdb::VectorGrid
		template <typename GridType>
		typename GridType::Ptr advect(const openvdb::FloatGrid::ConstPtr& grid,
		                              const openvdb::VectorGrid::ConstPtr& velocity, double dt);

		IncomingDataCache mNanoCache;
	};

   protected:
	// constructor, destructor
	SOP_VdbSolver(OP_Network* net, const char* name, OP_Operator* op);
	~SOP_VdbSolver() override;

	// labeling node inputs in Houdini UI
	const char* inputLabel(unsigned idx) const override;

	Cache cache;
};

}  // namespace VdbSolver

#endif  // __SOP_VDBSOLVER_HPP__