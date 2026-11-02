#include "duckdb/optimizer/join_order/mpf_plan_builder.hpp"

#include "duckdb/common/bit_utils.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

static unique_ptr<Expression> MPFColumnRef(const MPFColumn &column) {
	return make_uniq<BoundColumnRefExpression>(column.name, column.type, column.binding);
}

MPFPlanBuilder::MPFPlanBuilder(Optimizer &optimizer, const MPFQuery &query,
                               vector<unique_ptr<LogicalOperator>> relations)
    : optimizer(optimizer), query(query), relations(std::move(relations)) {
}

unique_ptr<Expression> MPFPlanBuilder::BuildProduct(const vector<MPFColumn> &values) {
	if (values.empty()) {
		return make_uniq<BoundConstantExpression>(Value::DOUBLE(1));
	}
	auto product = MPFColumnRef(values[0]);
	for (idx_t i = 1; i < values.size(); i++) {
		product = optimizer.BindScalarFunction("*", std::move(product), MPFColumnRef(values[i]));
	}
	return product;
}

MPFPlanBuilder::Output MPFPlanBuilder::BuildScan(const MPFPlanNode &node) {
	const auto relation_id = CountZeros<uint64_t>::Trailing(node.relations);
	auto &relation = query.relations[relation_id];
	Output result;
	result.op = std::move(relations[relation_id]);
	for (auto &variable : relation.variables) {
		for (auto &column : query.variables[variable].columns) {
			if (column.relation_id == relation_id) {
				result.variables.emplace(variable, column);
			}
		}
	}
	result.values = relation.measures;
	return result;
}

MPFPlanBuilder::Output MPFPlanBuilder::BuildJoin(const MPFPlanNode &node) {
	auto left = BuildNode(*node.left);
	auto right = BuildNode(*node.right);

	auto join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	Output result;
	result.variables = left.variables;
	for (auto &entry : right.variables) {
		auto match = left.variables.find(entry.first);
		if (match == left.variables.end()) {
			result.variables.insert(entry);
			continue;
		}
		JoinCondition condition;
		condition.left = MPFColumnRef(match->second);
		condition.right = MPFColumnRef(entry.second);
		condition.comparison = ExpressionType::COMPARE_EQUAL;
		join->conditions.push_back(std::move(condition));
	}
	D_ASSERT(!join->conditions.empty());
	result.values = left.values;
	result.values.insert(result.values.end(), right.values.begin(), right.values.end());
	join->children.push_back(std::move(left.op));
	join->children.push_back(std::move(right.op));
	result.op = std::move(join);
	return result;
}

MPFPlanBuilder::Output MPFPlanBuilder::BuildProject(const MPFPlanNode &node) {
	auto child = BuildNode(*node.left);
	const auto table_index = optimizer.binder.GenerateTableIndex();

	Output result;
	vector<unique_ptr<Expression>> select_list;
	for (auto &entry : child.variables) {
		if (node.output_variables & (mpf_set_t(1) << entry.first)) {
			auto column = entry.second;
			column.binding = ColumnBinding(table_index, select_list.size());
			result.variables.emplace(entry.first, std::move(column));
			select_list.push_back(MPFColumnRef(entry.second));
		}
	}
	const ColumnBinding value_binding(table_index, select_list.size());
	result.values.push_back({value_binding, LogicalType::DOUBLE, "product", DConstants::INVALID_INDEX});
	select_list.push_back(BuildProduct(child.values));

	auto projection = make_uniq<LogicalProjection>(table_index, std::move(select_list));
	projection->children.push_back(std::move(child.op));
	result.op = std::move(projection);
	return result;
}

MPFPlanBuilder::Output MPFPlanBuilder::BuildAggregate(const MPFPlanNode &node) {
	auto child = BuildNode(*node.left);
	const auto group_index = optimizer.binder.GenerateTableIndex();
	const auto aggregate_index = optimizer.binder.GenerateTableIndex();

	auto partial_sum = sum->Copy();
	partial_sum->Cast<BoundAggregateExpression>().children[0] = BuildProduct(child.values);
	vector<unique_ptr<Expression>> select_list;
	select_list.push_back(std::move(partial_sum));
	auto aggregate = make_uniq<LogicalAggregate>(group_index, aggregate_index, std::move(select_list));

	Output result;
	GroupingSet grouping_set;
	for (auto &entry : child.variables) {
		if (node.output_variables & (mpf_set_t(1) << entry.first)) {
			grouping_set.insert(aggregate->groups.size());
			auto column = entry.second;
			column.binding = ColumnBinding(group_index, aggregate->groups.size());
			result.variables.emplace(entry.first, std::move(column));
			aggregate->groups.push_back(MPFColumnRef(entry.second));
		}
	}
	aggregate->grouping_sets.push_back(std::move(grouping_set));
	result.values.push_back({ColumnBinding(aggregate_index, 0), LogicalType::DOUBLE, "sum", DConstants::INVALID_INDEX});

	aggregate->children.push_back(std::move(child.op));
	result.op = std::move(aggregate);
	return result;
}

MPFPlanBuilder::Output MPFPlanBuilder::BuildNode(const MPFPlanNode &node) {
	switch (node.type) {
	case MPFPlanNodeType::SCAN:
		return BuildScan(node);
	case MPFPlanNodeType::JOIN:
		return BuildJoin(node);
	case MPFPlanNodeType::PROJECT:
	case MPFPlanNodeType::OMITTED_AGGREGATE:
		return BuildProject(node);
	case MPFPlanNodeType::AGGREGATE:
		return BuildAggregate(node);
	default:
		throw InternalException("Unknown MPF plan node");
	}
}

unique_ptr<LogicalOperator> MPFPlanBuilder::Build(const MPFPlanNode &plan, LogicalAggregate &aggregate) {
	D_ASSERT(plan.type == MPFPlanNodeType::AGGREGATE);
	auto &query_sum = aggregate.expressions[0]->Cast<BoundAggregateExpression>();
	sum = query_sum;
	auto input = BuildNode(*plan.left);

	for (auto &group : aggregate.groups) {
		auto &column = group->Cast<BoundColumnRefExpression>();
		column.binding = input.variables.at(query.FindVariable(column.binding)).binding;
	}
	query_sum.children[0] = BuildProduct(input.values);
	return std::move(input.op);
}

} // namespace duckdb
