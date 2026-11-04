#include "duckdb/optimizer/join_order/mpf_column_facts.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"

namespace duckdb {

//! Whether this operator creates the table index
static bool IsProducer(const LogicalOperator &op, idx_t table_index) {
	const auto indexes = op.GetTableIndex();
	return std::find(indexes.begin(), indexes.end(), table_index) != indexes.end();
}

//! Whether the table index comes from `op` or from below it
static bool HasProducer(const LogicalOperator &op, idx_t table_index) {
	if (IsProducer(op, table_index)) {
		return true;
	}
	for (auto &child : op.children) {
		if (HasProducer(*child, table_index)) {
			return true;
		}
	}
	return false;
}

static MPFColumnFacts FactsOfColumn(ClientContext &context, const LogicalOperator &op, const ColumnBinding &column);

static MPFColumnFacts FactsOfBaseColumn(ClientContext &context, const LogicalGet &get, idx_t column_index) {
	MPFColumnFacts facts;
	auto &column_ids = get.GetColumnIds();
	if (!get.function.statistics_extended || column_index >= column_ids.size()) {
		return facts;
	}
	TableFunctionGetStatisticsInput input(get.bind_data.get(), column_ids[column_index]);
	auto statistics = get.function.statistics_extended(context, input);
	if (!statistics) {
		return facts;
	}
	facts.has_no_null = !statistics->CanHaveNull();
	if (statistics->GetStatsType() == StatisticsType::NUMERIC_STATS && NumericStats::HasMin(*statistics)) {
		const auto minimum = NumericStats::Min(*statistics).DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		facts.is_non_negative = minimum >= 0;
	}
	return facts;
}

static MPFColumnFacts FactsOfExpression(ClientContext &context, const LogicalOperator &input, const Expression &expr) {
	MPFColumnFacts facts;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF:
		return FactsOfColumn(context, input, expr.Cast<BoundColumnRefExpression>().binding);
	case ExpressionClass::BOUND_CONSTANT: {
		auto &value = expr.Cast<BoundConstantExpression>().value;
		if (!value.IsNull() && value.type().IsNumeric()) {
			const auto number = value.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
			facts.has_no_null = true;
			facts.is_non_negative = number >= 0;
		}
		return facts;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		// The planners multiply values in a Project
		auto &function = expr.Cast<BoundFunctionExpression>();
		if (function.function.name != "*" || function.children.size() != 2) {
			return facts;
		}
		const auto left = FactsOfExpression(context, input, *function.children[0]);
		const auto right = FactsOfExpression(context, input, *function.children[1]);
		facts.has_no_null = left.has_no_null && right.has_no_null;
		facts.is_non_negative = left.is_non_negative && right.is_non_negative;
		return facts;
	}
	case ExpressionClass::BOUND_AGGREGATE: {
		// A sum of numbers that are not NULL is not NULL, if there is at least one: the
		// aggregate checks that it has a GROUP BY, since without one an empty input still
		// gives a row
		auto &aggregate = expr.Cast<BoundAggregateExpression>();
		if (aggregate.function.name != "sum" || aggregate.children.size() != 1 || aggregate.filter) {
			return facts;
		}
		return FactsOfExpression(context, input, *aggregate.children[0]);
	}
	default:
		return facts;
	}
}

//! The facts of a column that comes from one of the children of the operator
static MPFColumnFacts FactsOfChildColumn(ClientContext &context, const LogicalOperator &op,
                                         const ColumnBinding &column) {
	for (auto &child : op.children) {
		if (HasProducer(*child, column.table_index)) {
			return FactsOfColumn(context, *child, column);
		}
	}
	return MPFColumnFacts();
}

//! Whether a join passes the values of this child as they are. An outer join pads the
//! columns of a child with NULL where a row of the other child has no partner.
static bool KeepsValuesOf(JoinType join_type, idx_t child) {
	switch (join_type) {
	case JoinType::INNER:
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::RIGHT_SEMI:
	case JoinType::RIGHT_ANTI:
	case JoinType::MARK:
		return true;
	case JoinType::LEFT:
	case JoinType::SINGLE:
		return child == 0;
	case JoinType::RIGHT:
		return child == 1;
	default:
		// OUTER pads both children
		return false;
	}
}

//! A join passes the columns of its children, unless it pads them. The output of an
//! inner equi-join has no NULL in the keys, and the keys are equal, so a key has what
//! its partner is known to have. A later optimizer can replace one key by the other, so
//! this is what makes the facts independent of which one a plan reads.
static MPFColumnFacts FactsOfJoinColumn(ClientContext &context, const LogicalJoin &join,
                                        const ColumnBinding &column) {
	MPFColumnFacts facts;
	for (idx_t child = 0; child < join.children.size(); child++) {
		if (HasProducer(*join.children[child], column.table_index)) {
			if (KeepsValuesOf(join.join_type, child)) {
				facts = FactsOfColumn(context, *join.children[child], column);
			}
			break;
		}
	}
	if (join.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || join.join_type != JoinType::INNER) {
		return facts;
	}
	for (auto &condition : join.Cast<LogicalComparisonJoin>().conditions) {
		if (condition.comparison != ExpressionType::COMPARE_EQUAL ||
		    condition.left->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
		    condition.right->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			continue;
		}
		const auto &left = condition.left->Cast<BoundColumnRefExpression>().binding;
		const auto &right = condition.right->Cast<BoundColumnRefExpression>().binding;
		if (left != column && right != column) {
			continue;
		}
		const auto partner = FactsOfChildColumn(context, join, left == column ? right : left);
		facts.has_no_null = true;
		facts.is_non_negative = facts.is_non_negative || partner.is_non_negative;
	}
	return facts;
}

//! The facts come from a list of operators that are known to keep or to make them. Any
//! other operator, including one of a later DuckDB, has none: it could pad a column
//! with NULL, as a POSITIONAL JOIN does, or compute it, as UNNEST does.
static MPFColumnFacts FactsOfColumn(ClientContext &context, const LogicalOperator &op, const ColumnBinding &column) {
	if (!IsProducer(op, column.table_index)) {
		switch (op.type) {
		// They drop rows, repeat them, or add columns, but pass the values of a column
		case LogicalOperatorType::LOGICAL_FILTER:
		case LogicalOperatorType::LOGICAL_ORDER_BY:
		case LogicalOperatorType::LOGICAL_TOP_N:
		case LogicalOperatorType::LOGICAL_LIMIT:
		case LogicalOperatorType::LOGICAL_SAMPLE:
		case LogicalOperatorType::LOGICAL_DISTINCT:
		case LogicalOperatorType::LOGICAL_WINDOW:
		case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
			return FactsOfChildColumn(context, op, column);
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
			return FactsOfJoinColumn(context, op.Cast<LogicalJoin>(), column);
		default:
			return MPFColumnFacts();
		}
	}
	auto producer = &op;
	switch (producer->type) {
	case LogicalOperatorType::LOGICAL_GET:
		return FactsOfBaseColumn(context, producer->Cast<LogicalGet>(), column.column_index);
	case LogicalOperatorType::LOGICAL_PROJECTION:
		if (column.column_index < producer->expressions.size()) {
			return FactsOfExpression(context, *producer->children[0], *producer->expressions[column.column_index]);
		}
		return MPFColumnFacts();
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggregate = producer->Cast<LogicalAggregate>();
		auto &input = *aggregate.children[0];
		// ROLLUP and the like turn groups into NULL
		if (aggregate.grouping_sets.size() > 1) {
			return MPFColumnFacts();
		}
		if (column.table_index == aggregate.group_index && column.column_index < aggregate.groups.size()) {
			return FactsOfExpression(context, input, *aggregate.groups[column.column_index]);
		}
		if (column.table_index == aggregate.aggregate_index && column.column_index < aggregate.expressions.size()) {
			auto facts = FactsOfExpression(context, input, *aggregate.expressions[column.column_index]);
			// Without GROUP BY, an empty input gives one row, with a NULL sum
			if (aggregate.groups.empty()) {
				facts.has_no_null = false;
			}
			return facts;
		}
		return MPFColumnFacts();
	}
	default:
		return MPFColumnFacts();
	}
}

MPFColumnFacts MPFColumnFacts::Of(ClientContext &context, const LogicalOperator &plan, const ColumnBinding &column) {
	// The join of a MatMul has not run yet: SPALM reads what comes from below it
	if (plan.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return FactsOfChildColumn(context, plan, column);
	}
	return FactsOfColumn(context, plan, column);
}

} // namespace duckdb
