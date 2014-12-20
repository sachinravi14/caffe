#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>

#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/size.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::string;
using std::max;

int size(string db_name) {
  // Lmdb
  MDB_env* mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_value;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;

  // Open db
  LOG(INFO) << "Opening lmdb " << db_name;
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(mdb_env_open(mdb_env, db_name.c_str(), MDB_RDONLY, 0664),
                  MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
          << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
          << "mdb_cursor_open failed";
  CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
                  MDB_SUCCESS);
  
  // Get size of training set
  LOG(INFO) << "Getting size of db ";
  int set_size = size(mdb_cursor, mdb_key, mdb_value); 
  
  return set_size;
}

int size(MDB_cursor* mdb_cursor, MDB_val mdb_key, MDB_val mdb_value) {
  Datum datum;
  int count = 0;
  do {
    // just a dummy operation
    datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
    //const string& data = datum.data();
    /*int size_in_datum = std::max<int>(datum.data().size(),
      datum.float_data_size());*/
    count++;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files";
    }
  } while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT)
                  == MDB_SUCCESS);
  return count; 
}
