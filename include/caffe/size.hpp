#ifndef SIZE_HPP_
#define SIZE_HPP_
#include <lmdb.h>

int size(MDB_cursor* mdb_cursor, MDB_val mdb_key, MDB_val mdb_value);

#endif 
