//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/mpf_plan_enumerator.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/optimizer/join_order/mpf_query.hpp"

namespace duckdb {

//! A set of relations or variables: bit i is the index i in MPFQuery
using mpf_set_t = uint64_t;

enum class MPFPlanNodeType : uint8_t { SCAN, JOIN, PROJECT, AGGREGATE };

//! A node of a candidate plan. There can be many candidates and only one runs,
//! so these are plain descriptions, not LogicalOperators.
struct MPFPlanNode {
	MPFPlanNodeType type;
	mpf_set_t relations;
	mpf_set_t output_variables;
	//! The output variables that are needed above this node: the query groups by
	//! them, or a relation that is not joined yet has them
	mpf_set_t required_variables;
	//! A Scan has one column per factor, and a Join has the columns of both sides.
	//! A Project or an Aggregate computes their product, which is one column.
	idx_t value_columns;
	//! The only child of a Project or an Aggregate is `left`
	optional_ptr<const MPFPlanNode> left;
	optional_ptr<const MPFPlanNode> right;
	//! Identifies the plan: S(r), J[v0](x y), P[v0,v1](x) and A[v0,v1](x)
	string description;
};

//! Enumerates all plans of an MPF query, based on "Including Group-By in Query
//! Optimization" (Chaudhuri and Shim) and on the CS+ algorithm of "Optimizing
//! MPF Queries". Without a cost model, every plan of a relation set is kept.
class MPFPlanEnumerator {
public:
	MPFPlanEnumerator(const MPFQuery &query, MPFPlannerType planner);

	//! Returns false and sets `reason` if the query has too many plans to keep
	bool Solve(string &reason);
	//! The complete plans, sorted by description, so that a query always gets the
	//! same numbering
	const vector<reference<const MPFPlanNode>> &Plans() const {
		return plans;
	}
	//! The plans as text; only the forced one, unless forced_plan is negative
	string ToString(int64_t forced_plan) const;

private:
	const MPFPlanNode &AddNode(MPFPlanNodeType type, mpf_set_t relations, mpf_set_t output_variables,
	                           idx_t value_columns, optional_ptr<const MPFPlanNode> left,
	                           optional_ptr<const MPFPlanNode> right, string description);
	const MPFPlanNode &AddScan(idx_t relation_id);
	const MPFPlanNode &AddJoin(const MPFPlanNode &left, const MPFPlanNode &right);
	const MPFPlanNode &AddProject(const MPFPlanNode &child);
	const MPFPlanNode &AddAggregate(const MPFPlanNode &child);
	//! A side of a MatMul must have one value column
	const MPFPlanNode &ProjectValues(const MPFPlanNode &side);
	//! Keeps a node of the relation set, and the aggregate above it if there can be one
	void Keep(vector<reference<const MPFPlanNode>> &set_plans, const MPFPlanNode &node);
	bool CanPushDownAggregate(const MPFPlanNode &node) const;
	string VariablesToString(mpf_set_t variables) const;

private:
	const MPFQuery &query;
	MPFPlannerType planner;
	mpf_set_t all_relations = 0;
	//! The variables of the GROUP BY
	mpf_set_t query_variables = 0;
	//! The relations that have a factor of the product
	mpf_set_t measure_relations = 0;
	//! The variables of each relation
	vector<mpf_set_t> relation_variables;
	//! The relation names in descriptions, which are unique unlike the names of a self join
	vector<string> relation_labels;
	vector<unique_ptr<MPFPlanNode>> nodes;
	vector<reference<const MPFPlanNode>> plans;
};

} // namespace duckdb
