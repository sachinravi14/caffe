#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/size.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::string;
using std::max;
using std::vector;

void write(int &adds, MDB_dbi mdb_dbi, MDB_txn* mdb_txn, MDB_env* mdb_env, MDB_val mdb_key, MDB_val mdb_value);
vector<string> readString(string &str);
/*
Code for splitting lmdb training set into two sets based on file with training item ids
*/

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  
  if (argc < 4) {
    LOG(ERROR) << "Usage: split_by_id file_to_supplement file_to_split id_file loc1/file1 loc2/file2";
    return 1;
  }
  
  MDB_val mdb_key, mdb_value;
   
  // Write entire file_to_supplement
  MDB_env* mdb_env_w0;
  MDB_dbi mdb_dbi_w0;
  MDB_txn* mdb_txn_w0;
  MDB_cursor* mdb_cursor_w0;
 
  CHECK_EQ(mdb_env_create(&mdb_env_w0), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_w0, 1099511627776), MDB_SUCCESS)  // 1TB
          << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env_w0, argv[1], 0, 0664), MDB_SUCCESS)
          << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_w0, NULL, 0, &mdb_txn_w0), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_w0, NULL, 0, &mdb_dbi_w0), MDB_SUCCESS)
          << "mdb_open failed. Does the lmdb already exist? ";
  CHECK_EQ(mdb_cursor_open(mdb_txn_w0, mdb_dbi_w0, &mdb_cursor_w0), MDB_SUCCESS)
          << "mdb_cursor_open failed";
  CHECK_EQ(mdb_cursor_get(mdb_cursor_w0, &mdb_key, &mdb_value, MDB_FIRST),
                  MDB_SUCCESS);
 
  // file1
  MDB_env* mdb_env_w1;
  MDB_dbi mdb_dbi_w1;
  MDB_txn* mdb_txn_w1;

  CHECK_EQ(mdb_env_create(&mdb_env_w1), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_w1, 1099511627776), MDB_SUCCESS)  // 1TB
          << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env_w1, argv[4], 0, 0664), MDB_SUCCESS)
          << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_w1, NULL, 0, &mdb_txn_w1), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_w1, NULL, 0, &mdb_dbi_w1), MDB_SUCCESS)
          << "mdb_open failed. Does the lmdb already exist? ";


  LOG(INFO) << "Writing all of file_to_supplement: " << argv[1]; 
  int numAdds1 = 0, numAdds2 = 0;
  do {
    // Write to file1
    write(numAdds1, mdb_dbi_w1, mdb_txn_w1, mdb_env_w1, mdb_key, mdb_value);  
  } while(mdb_cursor_get(mdb_cursor_w0, &mdb_key, &mdb_value, MDB_NEXT)
                  == MDB_SUCCESS);


  // Open file_to_split
  MDB_env* mdb_env;
  MDB_dbi mdb_dbi; 
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;

  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(mdb_env_open(mdb_env, argv[2], MDB_RDONLY, 0664),
                  MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
          << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
          << "mdb_cursor_open failed";
  CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
                  MDB_SUCCESS);  
 
  // file2
  MDB_env* mdb_env_w2;
  MDB_dbi mdb_dbi_w2;
  MDB_txn* mdb_txn_w2; 

  CHECK_EQ(mdb_env_create(&mdb_env_w2), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_w2, 1099511627776), MDB_SUCCESS)  // 1TB
          << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env_w2, argv[5], 0, 0664), MDB_SUCCESS)
          << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_w2, NULL, 0, &mdb_txn_w2), MDB_SUCCESS)
          << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_w2, NULL, 0, &mdb_dbi_w2), MDB_SUCCESS)
          << "mdb_open failed. Does the lmdb already exist? ";
 

  LOG(INFO) << "Reading ids from: " << argv[3];
  std::ifstream file(argv[3]);
  string line;
  int curr = 0;
  int status = -1;

  // Split file_to_split according to ids_file 
  while (getline(file, line)) {
    vector<string> str_list = readString(line);
    int num = atoi(str_list[0].c_str());
    int label = atoi(str_list[1].c_str());
    LOG(INFO) << "id: " << num << "; label: " << label;
     
    for (int i = curr; i < num; i++) { 
      // Write to file2
      write(numAdds2, mdb_dbi_w2, mdb_txn_w2, mdb_env_w2, mdb_key, mdb_value);  
      mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT);
     
      LOG(INFO) << "wrote " << i << " to file2: " << argv[5]; 
      ++curr;
    }

    // Write to file1 
    Datum datum;
    datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
    datum.set_label(label); 
  
    string value;
    datum.SerializeToString(&value); 
    mdb_value.mv_size = value.size();
    mdb_value.mv_data = reinterpret_cast<void*>(&value[0]);

    write(numAdds1, mdb_dbi_w1, mdb_txn_w1, mdb_env_w1, mdb_key, mdb_value);
    status = mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT);

    LOG(INFO) << "wrote " << curr << "to file1: " << argv[4];
    ++curr;
  }
  
  if (status == MDB_SUCCESS) {
    do {
      // Write rest of file_to_split to file2
      write(numAdds2, mdb_dbi_w2, mdb_txn_w2, mdb_env_w2, mdb_key, mdb_value); 
    }
    while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT)
                  == MDB_SUCCESS);
  }
  
  CHECK_EQ(mdb_txn_commit(mdb_txn_w1), MDB_SUCCESS) << "mdb_txn_commit failed"; 
  mdb_close(mdb_env_w1, mdb_dbi_w1);
  mdb_env_close(mdb_env_w1);  
 
  CHECK_EQ(mdb_txn_commit(mdb_txn_w2), MDB_SUCCESS) << "mdb_txn_commit failed"; 
  mdb_close(mdb_env_w2, mdb_dbi_w2);
  mdb_env_close(mdb_env_w2);
  
  LOG(INFO) << "Done.";
  return 0;
} 

void write(int &adds, MDB_dbi mdb_dbi, MDB_txn* mdb_txn, MDB_env* mdb_env, MDB_val mdb_key, MDB_val mdb_value) {
  CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_value, 0), MDB_SUCCESS)
          << "mdb_put failed";

  // Commit 
  if (adds % 1000 == 0) {
   CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
   CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
  }

  ++adds;
}

vector<string> readString(string &str) {
  vector<string> str_list;
  char c = ',';
  int i = 0, j = str.find(c);
  str_list.push_back(str.substr(i, j));
  str_list.push_back(str.substr(j+1, str.size() - 1));

  return str_list;
}
