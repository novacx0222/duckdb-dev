//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/mpf_column_facts.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/planner/column_binding.hpp"

namespace duckdb {

class ClientContext;
class LogicalOperator;

//! What the statistics of the base tables guarantee about a column of a plan. They can
//! be out of date, but only on the safe side: a deleted NULL still counts as a NULL.
//! Statistics exist only for base tables, so a column has no facts if it cannot be
//! traced to them, as with a window function or val + 1.
struct MPFColumnFacts {
	bool has_no_null = false;
	bool is_non_negative = false;

	//! The facts of a column in the output of `plan`. If `plan` is a join, these are the
	//! facts of its input, before the join.
	static MPFColumnFacts Of(ClientContext &context, const LogicalOperator &plan, const ColumnBinding &column);

	//! SPALM uses coordinates as indexes. NULL and negative numbers would stop the query.
	bool IsCoordinate() const {
		return has_no_null && is_non_negative;
	}
	//! SPALM skips a NULL value, and has no row for a group of them. Any other value is
	//! fine, whatever its sign: SPALM finds out at run time which pairs matched.
	bool IsValue() const {
		return has_no_null;
	}
};

} // namespace duckdb
