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
		facts.is_positive = minimum > 0;
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
			facts.is_positive = number > 0;
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
		facts.is_positive = left.is_positive && right.is_positive;
		facts.is_non_negative = facts.is_positive;
		return facts;
	}
	case ExpressionClass::BOUND_AGGREGATE: {
		// A group has at least one row, so a sum of positive numbers is positive
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

//! The output of an inner equi-join has no NULL in the keys, and the keys are equal, so
//! a key has what its partner is known to have. A later optimizer can replace one key
//! by the other, so this is what makes the facts independent of which one a plan reads.
static MPFColumnFacts FactsOfJoinColumn(ClientContext &context, const LogicalComparisonJoin &join,
                                        const ColumnBinding &column) {
	auto facts = FactsOfChildColumn(context, join, column);
	if (join.join_type != JoinType::INNER) {
		return facts;
	}
	for (auto &condition : join.conditions) {
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
		facts.is_positive = facts.is_positive || partner.is_positive;
	}
	return facts;
}

static MPFColumnFacts FactsOfColumn(ClientContext &context, const LogicalOperator &op, const ColumnBinding &column) {
	if (!IsProducer(op, column.table_index)) {
		if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			return FactsOfJoinColumn(context, op.Cast<LogicalComparisonJoin>(), column);
		}
		return FactsOfChildColumn(context, op, column);
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
			return FactsOfExpression(context, input, *aggregate.expressions[column.column_index]);
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
