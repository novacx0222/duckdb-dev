//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/mpf_query.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/planner/column_binding.hpp"

namespace duckdb {

class ClientContext;
class LogicalAggregate;
struct MPFHints;
class QueryGraphManager;

//! SET mpf_planner, which the SPALM extension registers
enum class MPFPlannerType : uint8_t { OFF, CS, CS_PLUS };

//! A column that the query reads from a relation
struct MPFColumn {
	ColumnBinding binding;
	LogicalType type;
	string name;
	//! The index in the RelationManager
	idx_t relation_id;
};

//! A variable of a marginalize-a-product-function query: the columns that an
//! inner equi-join makes equal, or one column in the GROUP BY.
struct MPFVariable {
	vector<MPFColumn> columns;
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
	vector<MPFColumn> measures;
	//! Whether an operator below can turn a group into NULL, which rules out a MatMul
	bool can_add_null_groups = false;
	//! The keys that hints declare, as indexes into MPFQuery::variables. A key
	//! only counts if the query joins or groups by all of its columns.
	vector<vector<idx_t>> unique_keys;
};

//! SELECT <groups>, SUM(m1 * ... * mn) over inner equi-joins of relations
struct MPFQuery {
	vector<MPFRelation> relations;
	vector<MPFVariable> variables;
	//! Whether there are hints: if so, a relation without a key can repeat rows
	bool has_hints = false;

	static MPFPlannerType GetPlanner(ClientContext &context);
	//! SET mpf_force_plan, which the SPALM extension registers, or -1 for no plan
	static int64_t GetForcedPlan(ClientContext &context);
	//! Reads the query from the aggregate above the join graph, which Build has
	//! extracted. Returns nullptr and sets `reason` unless the query is an MPF.
	static unique_ptr<MPFQuery> TryBuild(const LogicalAggregate &aggregate, QueryGraphManager &query_graph_manager,
	                                     const MPFHints &hints, string &reason);

	//! The index of the variable that has this column
	idx_t FindVariable(const ColumnBinding &binding) const;

	string ToString() const;
};

} // namespace duckdb
