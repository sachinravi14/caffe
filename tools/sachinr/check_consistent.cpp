#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include <float.h>
#include <vector>
#include <boost/algorithm/string.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
using std::vector;

/*
./check_consistent prefix
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);

  string image_list_filename = argv[1];
  image_list_filename += ".txt";
  string feature_filename = argv[1];
  feature_filename += ".b";
  std::ifstream image_list_file(image_list_filename.c_str());
  std::ifstream feature_file(feature_filename.c_str(), std::ios::in | std::ios::binary);

  string str;
  while (std::getline(image_list_file, str)) {
    float sum = 0;
    for (int i = 0; i < 205; ++i) {
      float p = 0;
      if (feature_file) {
        feature_file.read((char *)&p, sizeof(float));
      }
      else {
        LOG(ERROR) << "Finished file before expected.";
        return -1;
      }
      sum += p;
    }
    CHECK(sum <= 1.01);
    CHECK(sum >= 0.99);  
  }
  LOG(INFO) << "Consistent.";
  
  return 0;
}
