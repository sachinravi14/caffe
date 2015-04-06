#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <dirent.h>

#include <iomanip>
#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/size.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;
using caffe::Datum;
using caffe::BlobProto;
using std::string;
using std::vector;

string IMG_DIR = "/home/sachinr/exp-data/none-of-the-above-imgs/";

/*
./generate_none_of_the_above img_list db_name
Givne text file of image names + labels, generates lmdb
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);

  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  const char* db_path = argv[2];
  CHECK_EQ(mkdir(db_path, 0744), 0)
    << "mkdir " << db_path << "failed";
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
    << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
    << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
    << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
    << "mdb_open failed. Does the lmdb already exist?";

  string img_file = argv[1];
  std::ifstream file(img_file.c_str());
  string str;
  vector<string> parts;
  int count = 0;
  while (std::getline(file, str)) {
    boost::split(parts, str, boost::is_any_of(" "));
      
    Datum datum;
    parts[0] = IMG_DIR + parts[0];

    LOG(INFO) << "Writing file: " << parts[0];

    // Assign none-of-the-above image
    ReadImageToDatum(parts[0], 205, 256, 256, true, &datum);

    stringstream ss; 
    ss << std::setw(10) << std::setfill('0') << count;
    string id_prefix = ss.str(); 
   
    string value;
    datum.SerializeToString(&value);
    string keystr(id_prefix + "_" + parts[0]);
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    mdb_key.mv_size = keystr.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
      << "mdb_put failed";

    if (++count % 1000 == 0) {
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
        << "mdb_txn_commit failed";
      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    } 
  }

  if (++count % 1000 != 0) {
    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
  }

  return 0;
}
