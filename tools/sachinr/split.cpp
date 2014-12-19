#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

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
Code for splitting lmdb training set into two sets
*/
int AMOUNT = (int)2448873/2.0;
string PATH1 = "/data/sachinr/places-data/training_1";
string PATH2 = "/data/sachinr/places-data/training_2";

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  
  if (argc < 2) {
    LOG(ERROR) << "Usage: split input_file";
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
  
  MDB_env* mdb_env_w1;
  MDB_dbi mdb_dbi_w1;
  MDB_txn* mdb_txn_w1;

  MDB_env* mdb_env_w2;
  MDB_dbi mdb_dbi_w2;
  MDB_txn* mdb_txn_w2;

  CHECK_EQ(mkdir(PATH1.c_str(), 0744), 0)
        << "mkdir " << PATH1.c_str() << "failed";
  CHECK_EQ(mdb_env_create(&mdb_env_w1), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_w1, 1099511627776), MDB_SUCCESS)  // 1TB
          << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env_w1, PATH1.c_str(), 0, 0664), MDB_SUCCESS)
          << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_w1, NULL, 0, &mdb_txn_w1), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_w1, NULL, 0, &mdb_dbi_w1), MDB_SUCCESS)
          << "mdb_open failed. Does the lmdb already exist? ";

  CHECK_EQ(mkdir(PATH2.c_str(), 0744), 0)
        << "mkdir " << PATH1.c_str() << "failed";
  CHECK_EQ(mdb_env_create(&mdb_env_w2), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_w2, 1099511627776), MDB_SUCCESS)  // 1TB
          << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env_w2, PATH2.c_str(), 0, 0664), MDB_SUCCESS)
          << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_w2, NULL, 0, &mdb_txn_w2), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_w2, NULL, 0, &mdb_dbi_w2), MDB_SUCCESS)
          << "mdb_open failed. Does the lmdb already exist? ";

  int count = 0;
  do {
    if (count < AMOUNT) {
      CHECK_EQ(mdb_put(mdb_txn_w1, mdb_dbi_w1, &mdb_key, &mdb_value, 0), MDB_SUCCESS)
          << "mdb_put failed";
      
      if (count % 1000 == 0) {
        CHECK_EQ(mdb_txn_commit(mdb_txn_w1), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env_w1, NULL, 0, &mdb_txn_w1), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      }
    }
    else {
      CHECK_EQ(mdb_put(mdb_txn_w2, mdb_dbi_w2, &mdb_key, &mdb_value, 0), MDB_SUCCESS)
          << "mdb_put failed";
      if (count % 1000 == 0) {
        CHECK_EQ(mdb_txn_commit(mdb_txn_w2), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env_w2, NULL, 0, &mdb_txn_w2), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      }
    }
    count++; 
  } while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT)
                  == MDB_SUCCESS); 

  // Write last batch for both files
  CHECK_EQ(mdb_txn_commit(mdb_txn_w1), MDB_SUCCESS) << "mdb_txn_commit failed";
  mdb_close(mdb_env_w1, mdb_dbi_w1);
  mdb_env_close(mdb_env_w1);

  CHECK_EQ(mdb_txn_commit(mdb_txn_w2), MDB_SUCCESS) << "mdb_txn_commit failed";
  mdb_close(mdb_env_w2, mdb_dbi_w2);
  mdb_env_close(mdb_env_w2);
}


