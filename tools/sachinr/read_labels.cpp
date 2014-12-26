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

/*
Prints out item number & label for training set
*/

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  
  if (argc < 2) {
    LOG(ERROR) << "Usage: read_labels input_file";
    return 1;
  }

  // Lmdb
  MDB_env* mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_value;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;

  // Open db
  LOG(INFO) << "Opening lmdb " << argv[1];
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(mdb_env_open(mdb_env, argv[1], MDB_RDONLY, 0664),
                  MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
          << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
          << "mdb_cursor_open failed";
  CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
                  MDB_SUCCESS);
  int i = 0;
  do {
    Datum datum;
    datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
    LOG(INFO) << "item id: " << i << "; label: " << datum.label();
  
    ++i;
  } while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) == MDB_SUCCESS);


  return 0;
}
