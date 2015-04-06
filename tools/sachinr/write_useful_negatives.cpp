#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <sys/stat.h>

#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"

#include <iomanip>
#include <float.h>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <utility>
#include <omp.h>

using namespace caffe;
using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
using std::vector;

double threshold = 5.0;

const float UPPER_BOUND = 0.70;
const float LOWER_BOUND = 0.05;
const int TOP_K = 5;

const int INTERVAL = 10000;

const int NUM_THREADS = 15;

float get_entropy(vector<float> prob);

/*
From file prefixes, reads in image list file and binary file with feature probability info for
each image and selects images to write to lmdb based on criteria. 
./write_useful_negatives lmdb_name <list of file_prefixes>
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
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 2199023255552), MDB_SUCCESS)  // 1TB
    << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
    << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
    << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
    << "mdb_open failed. Does the lmdb already exist?";

  vector<std::pair<string, string> > image_feature_pairs;
  for (int i = 2; i < argc; ++i) {
    LOG(INFO) << "Prefix  " << (i-1) << ": " << argv[i];

    string image_list_filename = argv[i]; 
    image_list_filename += ".txt";
    string feature_filename = argv[i]; 
    feature_filename += ".b";
    image_feature_pairs.push_back(std::make_pair(image_list_filename, feature_filename));
  }
 
  vector<std::pair<string, string> > useful_data; 
  int total = 0, added = 0;
  for (int t = 0; t < image_feature_pairs.size(); ++t) { 
    std::ifstream image_list_file(image_feature_pairs[t].first.c_str());
    std::ifstream feature_file(image_feature_pairs[t].second.c_str(), std::ios::in | std::ios::binary);

    string str;
    while (std::getline(image_list_file, str)) {
      ++total;

      vector<float> prob;
      vector<std::pair<float, int> > bottom_data_vector;
      for (int i = 0; i < 205; ++i) {
        float p = 0;
        feature_file.read((char *)&p, sizeof(float));
        prob.push_back(p);
        bottom_data_vector.push_back(std::make_pair(p, i)); 
      }
    
      vector<string> parts; 
      boost::split(parts, str, boost::is_any_of(" "));
    
      int pos_label = -atoi(parts[1].c_str())-1;
      bool add = false;
    
      std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + TOP_K,
        bottom_data_vector.end(), std::greater<std::pair<float, int> >()); 

      for (int k = 0; k < TOP_K; k++) {
        if (bottom_data_vector[k].second == pos_label) {
          add = true;
          break;
        }
      }
     
      float entropy = get_entropy(prob);
      
      if (entropy > threshold || prob[pos_label] >= LOWER_BOUND) {
        add = true;
      } 

      if (prob[pos_label] >= UPPER_BOUND) {
        add = false;
      }

      // Write useful data to file
      if (add) {
        ++added;
        useful_data.push_back(std::make_pair(parts[0], parts[1]));
      //file << str << "\n"; 
      }

      if (total % INTERVAL == 0) {
        LOG(INFO) << "Total processed: " << total;
        LOG(INFO) << "Added item percentage: " << double(added)/total;
      } 
    }
    LOG(INFO) << "Finished file " << image_feature_pairs[t].first; 
  }   

  LOG(INFO) << "Total amount of data to be stored in lmdb: " << useful_data.size();
  shuffle(useful_data.begin(), useful_data.end());

  int batch_size = 1000;
  LOG(INFO) << "Number of iterations: " << useful_data.size()/batch_size;
  for (int i = 0; i < useful_data.size()/batch_size; ++i) {
    vector<string> key(batch_size);
    vector<Datum> vDatum(batch_size, Datum());
    #pragma omp parallel num_threads(NUM_THREADS)
    {
      #pragma omp for
      for (int j = 0; j < batch_size; ++j) {
        int indx = i * batch_size + j;
        std::pair<string, string> item = useful_data[indx];
        ReadImageToDatum(item.first.c_str(), atoi(item.second.c_str()), 256, 256, true, &vDatum[j]);
        
        // Need to generate sorted prefix so that can add random filenames as sorted keys  
        stringstream ss;
        ss << std::setw(10) << std::setfill('0') << indx;
        string id_prefix = ss.str();
        key[j] = id_prefix + "_" + item.first;
      }
    }
    LOG(INFO) << "Read Datums";

    for (int j = 0; j < batch_size; ++j) { 
      string value;
      Datum datum = vDatum[j];
      datum.SerializeToString(&value);
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = key[j].size();
      mdb_key.mv_data = reinterpret_cast<void*>(&key[j][0]);
      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
        << "mdb_put failed";
    }
    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
      << "mdb_txn_commit failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
    
    LOG(INFO) << "Finished batch " << (i+1) << "/" << useful_data.size()/batch_size; 
  } 


  int remainder = useful_data.size() % batch_size;
  int base_index = useful_data.size() - (remainder);
  vector<string> key(remainder);
  vector<Datum> vDatum(remainder, Datum());
  LOG(INFO) << "Finised all batches; " << remainder << " items remaining"; 
  #pragma omp parallel num_threads(NUM_THREADS) 
  {
    #pragma omp for
    for (int j = 0; j < remainder; ++j) {
      std::pair<std::string, std::string> item = useful_data[base_index + j];
      ReadImageToDatum(item.first.c_str(), atoi(item.second.c_str()), 256, 256, true, &vDatum[j]);

      // Need to generate sorted prefix so that can add random filenames as sorted keys  
      stringstream ss;
      ss << std::setw(10) << std::setfill('0') << base_index + j;
      string id_prefix = ss.str();
      key[j] = id_prefix + "_" + item.first; 
    }  
  }
  LOG(INFO) << "Read remainder datums"; 
  for (int j = 0; j < remainder; ++j) { 
    string value;
    Datum datum = vDatum[j];
    datum.SerializeToString(&value);
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    mdb_key.mv_size = key[j].size();
    mdb_key.mv_data = reinterpret_cast<void*>(&key[j][0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
      << "mdb_put failed";
  }
  CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
    << "mdb_txn_commit failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
    << "mdb_txn_begin failed";

  LOG(INFO) << "Finished."; 

  return 0;
}

float get_entropy(vector<float> prob) {
  float entropy = 0.0;
  for (int j = 0; j < prob.size(); ++j) {
    entropy -= prob[j] * (log(std::max(prob[j], float(FLT_MIN)))/(log(2.0)));
  }

  return entropy;
}
