//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/mpf_query.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/planner/column_binding.hpp"

namespace duckdb {

class ClientContext;
class LogicalAggregate;
class QueryGraphManager;

//! A variable of a marginalize-a-product-function query: the columns that an
//! inner equi-join makes equal, or one column in the GROUP BY.
struct MPFVariable {
	vector<ColumnBinding> columns;
	//! The columns as text, for example "r.k = s.row_id"
	string name;
	//! Whether the query groups by this variable
	bool is_query_variable = false;
};

//! A relation of the query. Everything below it stays as it is.
struct MPFRelation {
	//! The index in the RelationManager
	idx_t relation_id;
	string name;
	//! Indexes into MPFQuery::variables
	vector<idx_t> variables;
	//! The factors of the product read from this relation; a measure can repeat
	vector<ColumnBinding> measures;
	vector<string> measure_names;
};

//! SELECT <groups>, SUM(m1 * ... * mn) over inner equi-joins of relations
struct MPFQuery {
	vector<MPFRelation> relations;
	vector<MPFVariable> variables;

	//! Whether SET mpf_planner, which the SPALM extension registers, enables MPF planning
	static bool IsEnabled(ClientContext &context);
	//! Reads the query from the aggregate above the join graph, which Build has
	//! extracted. Returns nullptr and sets `reason` unless the query is an MPF.
	static unique_ptr<MPFQuery> TryBuild(const LogicalAggregate &aggregate, QueryGraphManager &query_graph_manager,
	                                     string &reason);

	string ToString() const;
};

} // namespace duckdb
