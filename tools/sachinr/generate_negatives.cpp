#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <dirent.h>

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

string FILE_PREFIX = "neg_category_";
string FILE_POSTFIX = ".txt";

string NFS_PATH = "/n/fs/lsun/imageTurk/";

string NEG_PREFIX = "neg_category_";

int MAX_CLASSES = 205;

vector<string> listFile(string path);
/*
Given specific negative class files, generate lmdb with negative instances
generate_negatives path_to_db_to_write images_dir neg_images_dir max_len <list of class files> 
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);

  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  const char* db_path = argv[1];
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

  int count = 0; 
  string images_dir = argv[2];
  string neg_images_dir = argv[3];
  int max_len = atoi(argv[4]);
  LOG(INFO) << "Image directory: " << images_dir;
  LOG(INFO) << "Negative Image directory: " << neg_images_dir;
  
  bool all = false;
  int actual_len = argc;
  vector<string> files;
  if (argc < 6) {
    LOG(INFO) << "No argument for negative class so processing all files...";
    all = true;
    files = listFile(neg_images_dir);
    actual_len = files.size() + 5;
  }

  std::vector<std::pair<std::string, std::string> > data;
  for (int i = 5; i < actual_len; i++) {
    // Read file and write images and corresponding labels to db 
    string filename;
    if (!all) {
      filename = neg_images_dir + FILE_PREFIX + argv[i] + FILE_POSTFIX;
      LOG(INFO) << "Processing class " << argv[i] << ": " << filename;
    }
    else {
      filename = neg_images_dir + files[i - 5];
      LOG(INFO) << "Processing: " << filename;
    }
    std::ifstream file(filename.c_str());
    string str;
   
    vector<string> parts;
    int items = 0;
    while (std::getline(file, str)) {
      boost::split(parts, str, boost::is_any_of(" "));
     
      parts[0].replace(0, NFS_PATH.size(), images_dir);
      data.push_back(std::make_pair(parts[0], parts[1])); 
      
      ++items;
      if (items > max_len) {
        break;
      } 
    }
  }  

  std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < data.size(); ++i) {
    Datum datum;
    ReadImageToDatum(data[i].first.c_str(), atoi(data[i].second.c_str()), 256, 256, true, &datum);
      
    string value;
    datum.SerializeToString(&value);
    string keystr(data[i].first);
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

vector<string> listFile(string path){
  DIR *pDIR;
  vector<string> files;
  struct dirent *entry;

  if( (pDIR=opendir(path.c_str())) ){
    while(entry = readdir(pDIR)) {
      if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ) {
        string str_name = string(entry->d_name);
        if (str_name.find(NEG_PREFIX) != std::string::npos) {
          files.push_back(str_name);
        }
      }
    }
    closedir(pDIR);
  }

  return files;
}
