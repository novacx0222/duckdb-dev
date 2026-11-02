#include "duckdb/optimizer/join_order/mpf_hints.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/qualified_name.hpp"

namespace duckdb {

//! Reads the text of a setting one token at a time
class MPFHintReader {
public:
	explicit MPFHintReader(const string &text) : text(text) {
	}

	bool AtEnd() {
		SkipSpaces();
		return position >= text.size();
	}

	string ReadName(const string &what) {
		SkipSpaces();
		const auto begin = position;
		while (position < text.size() && IsNameCharacter(text[position])) {
			position++;
		}
		if (position == begin) {
			throw Error("expected " + what);
		}
		return text.substr(begin, position - begin);
	}

	//! Whether the next character is this one, which it then skips
	bool TryRead(char character) {
		SkipSpaces();
		if (position >= text.size() || text[position] != character) {
			return false;
		}
		position++;
		return true;
	}

	InvalidInputException Error(const string &message) const {
		return InvalidInputException("mpf_hints: %s at position %llu of '%s'", message, position, text);
	}

private:
	static bool IsNameCharacter(char character) {
		return StringUtil::CharacterIsAlphaNumeric(character) || character == '_' || character == '.';
	}

	void SkipSpaces() {
		while (position < text.size() && StringUtil::CharacterIsSpace(text[position])) {
			position++;
		}
	}

	const string &text;
	idx_t position = 0;
};

MPFHints MPFHints::Parse(const string &text) {
	MPFHints hints;
	MPFHintReader reader(text);
	if (reader.AtEnd()) {
		return hints;
	}
	while (true) {
		MPFTableKey key;
		key.table = reader.ReadName("a table");
		if (!reader.TryRead('(')) {
			throw reader.Error("expected ( after table " + key.table);
		}
		do {
			auto column = StringUtil::Lower(reader.ReadName("a column of " + key.table));
			if (std::find(key.columns.begin(), key.columns.end(), column) != key.columns.end()) {
				throw reader.Error("column " + column + " is twice in the key of " + key.table);
			}
			key.columns.push_back(std::move(column));
		} while (reader.TryRead(','));
		if (!reader.TryRead(')')) {
			throw reader.Error("expected , or ) in the key of " + key.table);
		}
		hints.keys.push_back(std::move(key));
		if (reader.AtEnd()) {
			return hints;
		}
		if (!reader.TryRead(',')) {
			throw reader.Error("expected , after a key");
		}
	}
}

//! The table, or nullptr if it does not exist
static optional_ptr<TableCatalogEntry> LookUpTable(ClientContext &context, const MPFTableKey &key) {
	const auto name = QualifiedName::Parse(key.table);
	return Catalog::GetEntry<TableCatalogEntry>(context, name.catalog, name.schema, name.name,
	                                            OnEntryNotFound::RETURN_NULL);
}

TableCatalogEntry &MPFHints::GetTable(ClientContext &context, const MPFTableKey &key) {
	auto table = LookUpTable(context, key);
	if (!table) {
		throw InternalException("mpf_hints: table %s was not checked", key.table);
	}
	return *table;
}

void MPFHints::Validate(ClientContext &context) const {
	for (auto &key : keys) {
		auto table = LookUpTable(context, key);
		if (!table) {
			throw InvalidInputException("mpf_hints: table %s does not exist", key.table);
		}
		for (auto &column : key.columns) {
			if (!table->ColumnExists(column)) {
				throw InvalidInputException("mpf_hints: table %s has no column %s", key.table, column);
			}
		}
	}
}

MPFHints MPFHints::Get(ClientContext &context) {
	Value text;
	if (!context.TryGetCurrentSetting("mpf_hints", text) || text.IsNull()) {
		return MPFHints();
	}
	auto hints = Parse(text.ToString());
	hints.Validate(context);
	return hints;
}

} // namespace duckdb
