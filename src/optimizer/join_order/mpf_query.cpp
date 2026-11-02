#include "duckdb/optimizer/join_order/mpf_query.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/join_order/mpf_hints.hpp"
#include "duckdb/optimizer/join_order/query_graph_manager.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"

namespace duckdb {

MPFPlannerType MPFQuery::GetPlanner(ClientContext &context) {
	Value planner;
	if (!context.TryGetCurrentSetting("mpf_planner", planner) || planner.IsNull()) {
		return MPFPlannerType::OFF;
	}
	const auto name = StringUtil::Lower(planner.ToString());
	if (name == "cs") {
		return MPFPlannerType::CS;
	}
	return name == "cs_plus" ? MPFPlannerType::CS_PLUS : MPFPlannerType::OFF;
}

int64_t MPFQuery::GetForcedPlan(ClientContext &context) {
	Value plan;
	if (!context.TryGetCurrentSetting("mpf_force_plan", plan) || plan.IsNull()) {
		return -1;
	}
	return plan.GetValue<int64_t>();
}

//! Disjoint sets over consecutive indexes
struct MPFDisjointSets {
	vector<idx_t> parents;

	explicit MPFDisjointSets(idx_t count = 0) {
		for (idx_t i = 0; i < count; i++) {
			Add();
		}
	}
	idx_t Add() {
		parents.push_back(parents.size());
		return parents.size() - 1;
	}
	idx_t Find(idx_t index) {
		while (parents[index] != index) {
			parents[index] = parents[parents[index]];
			index = parents[index];
		}
		return index;
	}
	void Union(idx_t left, idx_t right) {
		parents[Find(right)] = Find(left);
	}
};

//! Collects the columns of a query until they are grouped into variables
class MPFColumns {
public:
	explicit MPFColumns(const unordered_map<idx_t, idx_t> &relation_mapping) : relation_mapping(relation_mapping) {
	}

	struct Column {
		MPFColumn column;
		//! With the relation, for example "r.k"
		string qualified_name;
		bool is_group;
	};

	vector<Column> columns;
	MPFDisjointSets sets;

	bool RelationOf(const ColumnBinding &binding, idx_t &relation_id) const {
		auto entry = relation_mapping.find(binding.table_index);
		if (entry == relation_mapping.end()) {
			return false;
		}
		relation_id = entry->second;
		return true;
	}

	//! The index of a column, added on its first use
	bool Add(const BoundColumnRefExpression &column, const vector<string> &relation_names, idx_t &index) {
		auto entry = indexes.find(column.binding);
		if (entry != indexes.end()) {
			index = entry->second;
			return true;
		}
		idx_t relation_id;
		if (!RelationOf(column.binding, relation_id)) {
			return false;
		}
		index = sets.Add();
		columns.push_back({{column.binding, column.return_type, column.GetName(), relation_id},
		                   relation_names[relation_id] + "." + column.GetName(),
		                   false});
		indexes[column.binding] = index;
		return true;
	}

	bool Contains(const ColumnBinding &binding) const {
		return indexes.find(binding) != indexes.end();
	}

private:
	const unordered_map<idx_t, idx_t> &relation_mapping;
	column_binding_map_t<idx_t> indexes;
};

//! Splits a product into its column factors
static bool ExtractFactors(const Expression &expression, vector<reference<const BoundColumnRefExpression>> &factors) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		factors.push_back(expression.Cast<BoundColumnRefExpression>());
		return true;
	}
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expression.Cast<BoundFunctionExpression>();
	if (function.function.name != "*" || function.children.size() != 2) {
		return false;
	}
	return ExtractFactors(*function.children[0], factors) && ExtractFactors(*function.children[1], factors);
}

//! The only aggregate must be a plain SUM over DOUBLE factors
static bool CheckAggregate(const LogicalAggregate &aggregate,
                           vector<reference<const BoundColumnRefExpression>> &factors, string &reason) {
	if (aggregate.grouping_sets.size() > 1 || !aggregate.grouping_functions.empty()) {
		reason = "grouping sets";
		return false;
	}
	if (aggregate.expressions.size() != 1 ||
	    aggregate.expressions[0]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
		reason = "not exactly one aggregate";
		return false;
	}
	auto &sum = aggregate.expressions[0]->Cast<BoundAggregateExpression>();
	if (sum.function.name != "sum" || sum.children.size() != 1) {
		reason = "not a SUM";
		return false;
	}
	if (sum.IsDistinct() || sum.filter || sum.order_bys) {
		reason = "SUM with DISTINCT, FILTER or ORDER BY";
		return false;
	}
	if (sum.return_type != LogicalType::DOUBLE) {
		reason = "not a DOUBLE SUM";
		return false;
	}
	if (!ExtractFactors(*sum.children[0], factors)) {
		reason = "SUM of something else than a product of columns";
		return false;
	}
	for (auto &group : aggregate.groups) {
		if (group->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			reason = "GROUP BY of something else than a column";
			return false;
		}
	}
	return true;
}

//! A relation is a table or a subquery. The others are left to the stock
//! planner: for example, a LEFT JOIN is a relation of the join graph.
static bool CheckRelation(const SingleJoinRelation &relation, string &reason) {
	reference<LogicalOperator> op = relation.op;
	while (op.get().type == LogicalOperatorType::LOGICAL_FILTER) {
		op = *op.get().children[0];
	}
	switch (op.get().type) {
	case LogicalOperatorType::LOGICAL_GET:
		if (op.get().children.empty()) {
			return true;
		}
		break;
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return true;
	default:
		break;
	}
	reason = "relation is a " + LogicalOperatorToString(op.get().type);
	return false;
}

//! An outer join pads its columns with NULL, and ROLLUP and the like drop groups.
//! SPALM cannot take a NULL row ID, so its optimizer checks the same operators.
static bool CanAddNullGroups(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		const auto join_type = op.Cast<LogicalJoin>().join_type;
		if (IsLeftOuterJoin(join_type) || IsRightOuterJoin(join_type) || join_type == JoinType::SINGLE) {
			return true;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		return true;
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		if (op.Cast<LogicalAggregate>().grouping_sets.size() > 1) {
			return true;
		}
		break;
	default:
		break;
	}
	for (auto &child : op.children) {
		if (CanAddNullGroups(*child)) {
			return true;
		}
	}
	return false;
}

//! An equi-join of two columns adds them to one variable, and a predicate on one
//! relation stays with this relation. Any other filter rules out the query.
static bool AddFilters(const vector<unique_ptr<FilterInfo>> &filters, const vector<string> &relation_names,
                       MPFColumns &columns, MPFDisjointSets &components, string &reason) {
	for (auto &filter_info : filters) {
		if (filter_info->join_type != JoinType::INNER) {
			reason = EnumUtil::ToString(filter_info->join_type) + " join";
			return false;
		}
		auto &set = filter_info->set.get();
		if (set.count == 1) {
			continue;
		}
		reason = "predicate over several relations: " + filter_info->filter->ToString();
		auto &filter = *filter_info->filter;
		if (set.count != 2 || filter.GetExpressionType() != ExpressionType::COMPARE_EQUAL) {
			return false;
		}
		auto &comparison = filter.Cast<BoundComparisonExpression>();
		if (comparison.left->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
		    comparison.right->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		idx_t left, right;
		if (!columns.Add(comparison.left->Cast<BoundColumnRefExpression>(), relation_names, left) ||
		    !columns.Add(comparison.right->Cast<BoundColumnRefExpression>(), relation_names, right)) {
			return false;
		}
		columns.sets.Union(left, right);
		components.Union(columns.columns[left].column.relation_id, columns.columns[right].column.relation_id);
		reason.clear();
	}
	return true;
}

//! Turns the disjoint sets of columns into variables
static bool BuildVariables(MPFColumns &columns, MPFQuery &query, string &reason) {
	unordered_map<idx_t, idx_t> variable_of_set;
	for (idx_t i = 0; i < columns.columns.size(); i++) {
		auto &column = columns.columns[i];
		const auto set = columns.sets.Find(i);
		auto entry = variable_of_set.find(set);
		if (entry == variable_of_set.end()) {
			entry = variable_of_set.emplace(set, query.variables.size()).first;
			query.variables.emplace_back();
		}
		const auto variable_index = entry->second;
		auto &variable = query.variables[variable_index];
		if (!variable.columns.empty()) {
			variable.name += " = ";
		}
		variable.name += column.qualified_name;
		variable.columns.push_back(column.column);
		if (column.is_group) {
			if (variable.is_query_variable) {
				reason = "GROUP BY of two columns of one variable: " + variable.name;
				return false;
			}
			variable.is_query_variable = true;
		}
		auto &relation = query.relations[column.column.relation_id];
		for (auto &relation_variable : relation.variables) {
			if (relation_variable == variable_index) {
				reason = "two columns of one relation in a variable: " + variable.name;
				return false;
			}
		}
		relation.variables.push_back(variable_index);
	}
	return true;
}

//! The scan of a relation that is a base table, possibly below filters
static optional_ptr<const LogicalGet> FindBaseTable(const LogicalOperator &relation) {
	reference<const LogicalOperator> op = relation;
	while (op.get().type == LogicalOperatorType::LOGICAL_FILTER) {
		op = *op.get().children[0];
	}
	if (op.get().type != LogicalOperatorType::LOGICAL_GET) {
		return nullptr;
	}
	return op.get().Cast<LogicalGet>();
}

//! The name of a column in its table, which an alias in the query does not change
static string ColumnNameInTable(const LogicalGet &get, const MPFColumn &column) {
	const auto index =
	    get.projection_ids.empty() ? column.binding.column_index : get.projection_ids[column.binding.column_index];
	return StringUtil::Lower(get.GetColumnName(get.GetColumnIds()[index]));
}

//! Turns the hints on the table of a relation into keys over variables of the query
static void AddUniqueKeys(ClientContext &context, const MPFHints &hints, const LogicalOperator &relation_op,
                          idx_t relation_id, MPFQuery &query) {
	auto get = FindBaseTable(relation_op);
	auto table = get ? get->GetTable() : nullptr;
	if (!table) {
		return;
	}
	unordered_map<string, idx_t> variable_of_column;
	auto &relation = query.relations[relation_id];
	for (auto &variable : relation.variables) {
		for (auto &column : query.variables[variable].columns) {
			if (column.relation_id == relation_id) {
				variable_of_column[ColumnNameInTable(*get, column)] = variable;
			}
		}
	}
	for (auto &key : hints.keys) {
		if (&MPFHints::GetTable(context, key) != table.get()) {
			continue;
		}
		vector<idx_t> variables;
		for (auto &column : key.columns) {
			auto entry = variable_of_column.find(column);
			if (entry != variable_of_column.end()) {
				variables.push_back(entry->second);
			}
		}
		if (variables.size() == key.columns.size()) {
			relation.unique_keys.push_back(std::move(variables));
		}
	}
}

unique_ptr<MPFQuery> MPFQuery::TryBuild(const LogicalAggregate &aggregate, QueryGraphManager &query_graph_manager,
                                        const MPFHints &hints, string &reason) {
	vector<reference<const BoundColumnRefExpression>> factors;
	if (!CheckAggregate(aggregate, factors, reason)) {
		return nullptr;
	}

	auto query = make_uniq<MPFQuery>();
	auto &relation_manager = query_graph_manager.relation_manager;
	vector<string> relation_names;
	auto &relations = relation_manager.Relations();
	for (idx_t i = 0; i < relations.size(); i++) {
		if (!CheckRelation(*relations[i], reason)) {
			return nullptr;
		}
		auto name = relations[i]->stats.table_name;
		relation_names.push_back(name.empty() ? "R" + to_string(i) : name);
		query->relations.emplace_back();
		query->relations.back().relation_id = i;
		query->relations.back().name = relation_names.back();
		query->relations.back().can_add_null_groups = CanAddNullGroups(relations[i]->op);
	}

	MPFColumns columns(relation_manager.relation_mapping);
	MPFDisjointSets components(relations.size());
	if (!AddFilters(query_graph_manager.GetFilterBindings(), relation_names, columns, components, reason)) {
		return nullptr;
	}
	for (idx_t i = 1; i < relations.size(); i++) {
		if (components.Find(i) != components.Find(0)) {
			reason = "cross product";
			return nullptr;
		}
	}

	for (auto &group : aggregate.groups) {
		idx_t index;
		if (!columns.Add(group->Cast<BoundColumnRefExpression>(), relation_names, index)) {
			reason = "GROUP BY of a column outside the relations";
			return nullptr;
		}
		columns.columns[index].is_group = true;
	}
	if (!BuildVariables(columns, *query, reason)) {
		return nullptr;
	}

	for (auto &factor : factors) {
		auto &measure = factor.get();
		idx_t relation_id;
		if (!columns.RelationOf(measure.binding, relation_id)) {
			reason = "measure outside the relations";
			return nullptr;
		}
		if (columns.Contains(measure.binding)) {
			reason = "measure is a variable: " + measure.GetName();
			return nullptr;
		}
		query->relations[relation_id].measures.push_back(
		    {measure.binding, measure.return_type, measure.GetName(), relation_id});
	}

	query->has_hints = !hints.keys.empty();
	if (query->has_hints) {
		for (idx_t i = 0; i < relations.size(); i++) {
			AddUniqueKeys(query_graph_manager.context, hints, relations[i]->op, i, *query);
		}
	}
	return query;
}

idx_t MPFQuery::FindVariable(const ColumnBinding &binding) const {
	for (idx_t i = 0; i < variables.size(); i++) {
		for (auto &column : variables[i].columns) {
			if (column.binding == binding) {
				return i;
			}
		}
	}
	throw InternalException("MPF query: column is not in a variable");
}

string MPFQuery::ToString() const {
	string result =
	    "[mpf] accepted: " + to_string(relations.size()) + " relations, " + to_string(variables.size()) + " variables";
	for (auto &relation : relations) {
		vector<string> names;
		for (auto &variable : relation.variables) {
			names.push_back("v" + to_string(variable));
		}
		vector<string> measure_names;
		for (auto &measure : relation.measures) {
			measure_names.push_back(measure.name);
		}
		result += "\n[mpf]   " + relation.name + "(" + StringUtil::Join(names, ", ") + ") measures [" +
		          StringUtil::Join(measure_names, ", ") + "]";
		if (has_hints) {
			vector<string> keys;
			for (auto &key : relation.unique_keys) {
				vector<string> key_names;
				for (auto &variable : key) {
					key_names.push_back("v" + to_string(variable));
				}
				keys.push_back("(" + StringUtil::Join(key_names, ", ") + ")");
			}
			result += " keys [" + StringUtil::Join(keys, ", ") + "]";
		}
	}
	for (idx_t i = 0; i < variables.size(); i++) {
		result += "\n[mpf]   v" + to_string(i) + ": " + variables[i].name;
		if (variables[i].is_query_variable) {
			result += " (query)";
		}
	}
	return result;
}

} // namespace duckdb
