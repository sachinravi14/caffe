#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <dirent.h>

#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <boost/lexical_cast.hpp>

using namespace caffe;
using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
using std::vector;

const int NUM_CHANNELS = 3;
const int NUM_ROWS = 256;
const int NUM_COLS = 256;
const int LABEL = 205; 

const string IMG_PREFIX = "random_img";

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

  int num_images = atoi(argv[1]);
 
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);

  int count = 0; 
  for (int i = 0; i < num_images; ++i) {
    cv::Mat random(NUM_ROWS, NUM_COLS, CV_8UC3);
    cv::randn(random, cv::Scalar::all(128), cv::Scalar::all(10));
    string img_name = IMG_PREFIX + boost::lexical_cast<std::string>(i) + ".jpg";
    //cv::imwrite(img_name, random, compression_params); 
  
    Datum datum;
    datum.set_channels(NUM_CHANNELS);
    datum.set_height(NUM_ROWS);
    datum.set_width(NUM_COLS);
    datum.set_label(LABEL);
    datum.clear_data();
    datum.clear_float_data();
    string* datum_string = datum.mutable_data();
    for (int c = 0; c < NUM_CHANNELS; ++c) {
      for (int h = 0; h < random.rows; ++h) {
        for (int w = 0; w < random.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(random.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }

   
    string value;
    datum.SerializeToString(&value);
    string keystr(img_name);
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
