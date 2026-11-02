//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/mpf_hints.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

class ClientContext;
class TableCatalogEntry;

//! A unique key that a hint declares: the rows of the table differ on these columns
struct MPFTableKey {
	//! As written in the hint, which can have a schema and a catalog
	string table;
	//! In lower case, like the comparison of identifiers
	vector<string> columns;
};

//! SET mpf_hints = 'table(column, ...), ...', which the SPALM extension registers.
//! DuckDB does not know which tables repeat rows, so the MPF planners are told.
//! Without hints, they add no aggregate only to merge repeated rows. With hints, a
//! table without a key can repeat rows, and the keys of a plan are inferred from them.
//! A hint can only change the candidate plans: all of them return the query's result,
//! so nothing checks that the rows really differ.
struct MPFHints {
	vector<MPFTableKey> keys;

	//! Throws InvalidInputException unless the text has the form above
	static MPFHints Parse(const string &text);
	//! Throws InvalidInputException if a table or a column does not exist
	void Validate(ClientContext &context) const;
	//! The hints of the connection, after checking them
	static MPFHints Get(ClientContext &context);
	//! The table of a key, which must exist
	static TableCatalogEntry &GetTable(ClientContext &context, const MPFTableKey &key);
};

} // namespace duckdb
