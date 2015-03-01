#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

#include <algorithm>
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/size.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::string;
using std::vector;

float CUTOFF = 0.70;
string postfix = "_pruned"; 

/*
Given test lmdb and softmax output txt file, generate new lmdb with high confidence 
items for negative class removed.
./remove_high_conf db_to_prune softmax_output_file
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);

  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_value;
  MDB_txn *mdb_txn;
  MDB_cursor* mdb_cursor;
 
  LOG(INFO) << "Starting...";
 
  const char* db_path = argv[1];
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
    << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
    << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
          << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
          << "mdb_cursor_open failed";
  CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
                  MDB_SUCCESS);


  LOG(INFO) << "Next step...";

  MDB_env *new_mdb_env;
  MDB_dbi new_mdb_dbi;
  MDB_txn *new_mdb_txn;
  
  string new_name = argv[1] + postfix;
  db_path = new_name.c_str();
  CHECK_EQ(mkdir(db_path, 0744), 0)
    << "mkdir " << db_path << "failed";
  CHECK_EQ(mdb_env_create(&new_mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(new_mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
    << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(new_mdb_env, db_path, 0, 0664), MDB_SUCCESS)
    << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(new_mdb_env, NULL, 0, &new_mdb_txn), MDB_SUCCESS)
    << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(new_mdb_txn, NULL, 0, &new_mdb_dbi), MDB_SUCCESS)
    << "mdb_open failed. Does the lmdb already exist?";


  LOG(INFO) << "Processing softmax...";

  // Read softmax file
  int count = 0;
  string str;
  std::ifstream file(argv[2]);
  vector<string> parts;
  while (std::getline(file, str)) {
    boost::split(parts, str, boost::is_any_of(" "));
    LOG(INFO) << "Key: " << parts[0];
    LOG(INFO) << "Label: " << parts[1];
    int label = atoi(parts[1].c_str());
    vector<float> prob;
    float true_label_prob = atof(parts[2 + label].c_str());
    LOG(INFO) << "Prob: " << true_label_prob;
    if (true_label_prob < CUTOFF) {
      mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT);
      CHECK_EQ(mdb_put(new_mdb_txn, new_mdb_dbi, &mdb_key, &mdb_value, 0), MDB_SUCCESS)
        << "mdb_put failed";

      if (++count % 1000 == 0) {
        CHECK_EQ(mdb_txn_commit(new_mdb_txn), MDB_SUCCESS)
          << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(new_mdb_env, NULL, 0, &new_mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      }
    }
    
  }

  if (++count % 1000 != 0) {
    CHECK_EQ(mdb_txn_commit(new_mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(new_mdb_env, new_mdb_dbi);
    mdb_env_close(new_mdb_env);
  }

  return 0;
}
