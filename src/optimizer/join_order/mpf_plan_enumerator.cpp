#include "duckdb/optimizer/join_order/mpf_plan_enumerator.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

//! Relation sets are visited one by one, and each set is split in all ways
static constexpr idx_t MPF_MAX_RELATIONS = 12;
static constexpr idx_t MPF_MAX_VARIABLES = 64;
//! Plans of a query share nodes, but their number still grows very fast
static constexpr idx_t MPF_MAX_PLAN_NODES = 1000000;

static mpf_set_t MPFSetOf(idx_t index) {
	return mpf_set_t(1) << index;
}

static idx_t MPFSetSize(mpf_set_t set) {
	idx_t size = 0;
	for (; set; set &= set - 1) {
		size++;
	}
	return size;
}

MPFPlanEnumerator::MPFPlanEnumerator(const MPFQuery &query, MPFPlannerType planner) : query(query), planner(planner) {
}

const MPFPlanNode &MPFPlanEnumerator::AddNode(MPFPlanNodeType type, mpf_set_t relations, mpf_set_t output_variables,
                                              idx_t value_columns, optional_ptr<const MPFPlanNode> left,
                                              optional_ptr<const MPFPlanNode> right, string description) {
	mpf_set_t future_variables = 0;
	for (idx_t i = 0; i < relation_variables.size(); i++) {
		if (!(relations & MPFSetOf(i))) {
			future_variables |= relation_variables[i];
		}
	}
	auto node = make_uniq<MPFPlanNode>();
	node->type = type;
	node->relations = relations;
	node->output_variables = output_variables;
	node->required_variables = output_variables & (query_variables | future_variables);
	node->value_columns = value_columns;
	node->left = left;
	node->right = right;
	node->description = std::move(description);
	nodes.push_back(std::move(node));
	return *nodes.back();
}

string MPFPlanEnumerator::VariablesToString(mpf_set_t variables) const {
	vector<string> names;
	for (idx_t i = 0; i < query.variables.size(); i++) {
		if (variables & MPFSetOf(i)) {
			names.push_back("v" + to_string(i));
		}
	}
	// Sorted as text, the same way as the sides of a join
	std::sort(names.begin(), names.end());
	return "[" + StringUtil::Join(names, ",") + "]";
}

const MPFPlanNode &MPFPlanEnumerator::AddScan(idx_t relation_id) {
	return AddNode(MPFPlanNodeType::SCAN, MPFSetOf(relation_id), relation_variables[relation_id],
	               query.relations[relation_id].measures.size(), nullptr, nullptr,
	               "S(" + relation_labels[relation_id] + ")");
}

const MPFPlanNode &MPFPlanEnumerator::AddJoin(const MPFPlanNode &left, const MPFPlanNode &right) {
	const auto &first = left.description < right.description ? left.description : right.description;
	const auto &second = left.description < right.description ? right.description : left.description;
	return AddNode(
	    MPFPlanNodeType::JOIN, left.relations | right.relations, left.output_variables | right.output_variables,
	    left.value_columns + right.value_columns, &left, &right,
	    "J" + VariablesToString(left.output_variables & right.output_variables) + "(" + first + " " + second + ")");
}

const MPFPlanNode &MPFPlanEnumerator::AddProject(const MPFPlanNode &child) {
	// No rows are merged, so the other variables can go
	return AddNode(MPFPlanNodeType::PROJECT, child.relations, child.required_variables, 1, &child, nullptr,
	               "P" + VariablesToString(child.required_variables) + "(" + child.description + ")");
}

const MPFPlanNode &MPFPlanEnumerator::ProjectValues(const MPFPlanNode &side) {
	return side.value_columns == 1 ? side : AddProject(side);
}

//! Whether an aggregate grouping by `groups` above the node has the shape of a
//! MatMul: C(row, column) = SUM(A(row, k) * B(k, column)). A coordinate may
//! repeat, since SPALM can sum the entries in the CSR (SET spalm_duplicates).
static bool IsMatMul(mpf_set_t groups, const MPFPlanNode &node) {
	if (node.type != MPFPlanNodeType::JOIN) {
		return false;
	}
	if (MPFSetSize(node.left->output_variables & node.right->output_variables) != 1) {
		return false;
	}
	// A join variable is in both sides, so with one group per side it is summed out
	return MPFSetSize(groups) == 2 && MPFSetSize(groups & node.left->output_variables) == 1 &&
	       MPFSetSize(groups & node.right->output_variables) == 1;
}

const MPFPlanNode &MPFPlanEnumerator::AddAggregate(const MPFPlanNode &child) {
	// The groups are all the variables that are still needed, so the aggregate
	// removes the others
	reference<const MPFPlanNode> input = child;
	if (IsMatMul(child.required_variables, child)) {
		auto &left = ProjectValues(*child.left);
		auto &right = ProjectValues(*child.right);
		if (&left != child.left.get() || &right != child.right.get()) {
			input = AddJoin(left, right);
		}
	}
	return AddNode(MPFPlanNodeType::AGGREGATE, child.relations, child.required_variables, 1, input.get(), nullptr,
	               "A" + VariablesToString(child.required_variables) + "(" + input.get().description + ")");
}

bool MPFPlanEnumerator::CanPushDownAggregate(const MPFPlanNode &node) const {
	// An aggregate that removes no variables merges nothing
	if (!(node.output_variables & ~node.required_variables)) {
		return false;
	}
	if (planner == MPFPlannerType::CS_PLUS) {
		return true;
	}
	// Simple coalescing grouping: the node has all the factors of the product.
	// CS+ only relies on SUM distributing over the join.
	return (node.relations & measure_relations) == measure_relations;
}

void MPFPlanEnumerator::Keep(vector<reference<const MPFPlanNode>> &set_plans, const MPFPlanNode &node) {
	set_plans.push_back(node);
	// The aggregate of the complete plan is the one of the query
	if (node.relations != all_relations && CanPushDownAggregate(node)) {
		set_plans.push_back(AddAggregate(node));
	}
}

bool MPFPlanEnumerator::Solve(string &reason) {
	const auto relation_count = query.relations.size();
	if (relation_count > MPF_MAX_RELATIONS) {
		reason = to_string(relation_count) + " relations";
		return false;
	}
	if (query.variables.size() > MPF_MAX_VARIABLES) {
		reason = to_string(query.variables.size()) + " variables";
		return false;
	}

	all_relations = MPFSetOf(relation_count) - 1;
	for (idx_t i = 0; i < query.variables.size(); i++) {
		if (query.variables[i].is_query_variable) {
			query_variables |= MPFSetOf(i);
		}
	}
	for (idx_t i = 0; i < relation_count; i++) {
		auto &relation = query.relations[i];
		mpf_set_t variables = 0;
		for (auto &variable : relation.variables) {
			variables |= MPFSetOf(variable);
		}
		relation_variables.push_back(variables);
		if (!relation.measures.empty()) {
			measure_relations |= MPFSetOf(i);
		}
		auto label = relation.name;
		for (idx_t other = 0; other < relation_count; other++) {
			if (other != i && query.relations[other].name == relation.name) {
				label += "#" + to_string(i);
				break;
			}
		}
		relation_labels.push_back(std::move(label));
	}

	// A subset is smaller than its set as a number, so its plans are done first
	vector<vector<reference<const MPFPlanNode>>> set_plans(all_relations + 1);
	for (idx_t i = 0; i < relation_count; i++) {
		Keep(set_plans[MPFSetOf(i)], AddScan(i));
	}
	for (mpf_set_t set = 1; set <= all_relations; set++) {
		if (MPFSetSize(set) == 1) {
			continue;
		}
		// Joins are commutative, so the left side always has the lowest relation
		const auto lowest = set & (~set + 1);
		const auto others = set ^ lowest;
		auto left_others = (others - 1) & others;
		while (true) {
			const auto left_set = lowest | left_others;
			for (auto &left : set_plans[left_set]) {
				for (auto &right : set_plans[set ^ left_set]) {
					// No shared variable would be a cross join
					if (left.get().output_variables & right.get().output_variables) {
						Keep(set_plans[set], AddJoin(left, right));
					}
				}
				if (nodes.size() > MPF_MAX_PLAN_NODES) {
					reason = "more than " + to_string(MPF_MAX_PLAN_NODES) + " plan nodes";
					return false;
				}
			}
			if (left_others == 0) {
				break;
			}
			left_others = (left_others - 1) & others;
		}
	}

	for (auto &plan : set_plans[all_relations]) {
		plans.push_back(AddAggregate(plan));
	}
	std::sort(plans.begin(), plans.end(),
	          [](const MPFPlanNode &left, const MPFPlanNode &right) { return left.description < right.description; });
	return true;
}

string MPFPlanEnumerator::ToString(int64_t forced_plan) const {
	const auto plan_count = plans.size();
	string result = "[mpf] plans: " + to_string(plan_count);
	if (forced_plan >= 0 && idx_t(forced_plan) >= plan_count) {
		return result + ", no plan " + to_string(forced_plan);
	}
	for (idx_t i = 0; i < plan_count; i++) {
		if (forced_plan < 0 || idx_t(forced_plan) == i) {
			result += "\n[mpf] plan " + to_string(i) + ": " + plans[i].get().description;
		}
	}
	return result;
}

} // namespace duckdb
