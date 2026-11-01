//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/mpf_plan_builder.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/optimizer/join_order/mpf_plan_enumerator.hpp"
#include "duckdb/optimizer/join_order/mpf_query.hpp"

namespace duckdb {

class Expression;
class LogicalAggregate;
class LogicalOperator;
class Optimizer;

//! Turns a candidate plan of an MPF query into logical operators
class MPFPlanBuilder {
public:
	//! The relations are indexed as in the query, and take the predicates on one relation with them
	MPFPlanBuilder(Optimizer &optimizer, const MPFQuery &query, vector<unique_ptr<LogicalOperator>> relations);

	//! Returns the operators below the root of the plan, which is the aggregate of
	//! the query. An aggregate below can replace the columns that it reads, and it
	//! also computes part of the product, so the aggregate of the query is rewritten.
	unique_ptr<LogicalOperator> Build(const MPFPlanNode &plan, LogicalAggregate &aggregate);

private:
	//! What a node outputs. Above an aggregate or a projection, a variable and the
	//! values are other columns than the ones of the relations.
	struct Output {
		unique_ptr<LogicalOperator> op;
		//! The column of each variable, by its index in the query. A join keeps
		//! one column of a variable: the other ones are equal.
		map<idx_t, MPFColumn> variables;
		vector<MPFColumn> values;
	};

	Output BuildNode(const MPFPlanNode &node);
	Output BuildScan(const MPFPlanNode &node);
	Output BuildJoin(const MPFPlanNode &node);
	Output BuildProject(const MPFPlanNode &node);
	Output BuildAggregate(const MPFPlanNode &node);
	//! The product of the values; 1 if there are none, since it is a neutral factor
	unique_ptr<Expression> BuildProduct(const vector<MPFColumn> &values);

private:
	Optimizer &optimizer;
	const MPFQuery &query;
	vector<unique_ptr<LogicalOperator>> relations;
	//! The SUM of the query, which the aggregates of a plan copy
	optional_ptr<const Expression> sum;
};

} // namespace duckdb
