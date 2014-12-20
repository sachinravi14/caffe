#ifndef SIZE_HPP_
#define SIZE_HPP_

#include <lmdb.h>
#include <string>

using std::string;

int size(string db_name);

int size(MDB_cursor* mdb_cursor, MDB_val mdb_key, MDB_val mdb_value);

#endif 
