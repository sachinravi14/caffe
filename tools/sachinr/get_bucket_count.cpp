#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <boost/unordered_map.hpp>
#include <boost/lexical_cast.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/size.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::string;
using std::max;

int MIN_LABEL = -300;
int MAX_LABEL = 300;

/*
Prints out label & corresponding frequency for training set
./get_bucket_count lmdb max_amt_to_read(optional)
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
 
  typedef boost::unordered_map<std::string, int> freq_map;
  freq_map map;
 
  int max = 0;
  if (argc < 2) {
    LOG(ERROR) << "Usage: read_labels input_file max_read";
    return 1;
  }
  if (argc < 3) {
    max = INT_MAX;
  }
  else {
    max = atoi(argv[2]);
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
  double total = 0;
  do {
    Datum datum;
    datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
 
    int label_int = datum.label();
    string label = boost::lexical_cast<string>(label_int);
    //datum.label().SerializeToString(&label);
    int count = 0;
    if (map.count(label) != 0) {
      count = map.at(label);
    }
    
    map.erase(label);
    map.insert(freq_map::value_type(label, count + 1));

    ++total;
    if (total >= max) {
      break;
    }
  } while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) == MDB_SUCCESS);


  for (int i = MIN_LABEL; i < MAX_LABEL; ++i) {
    string label = boost::lexical_cast<string>(i);
    if (map.count(label) != 0) {
      LOG(INFO) << label << ": " << map.at(label)/total;
    }
  }

  return 0;
}
