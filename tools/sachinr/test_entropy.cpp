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

float get_entropy(vector<float> &prob);

/*
Given entropy and softmax output txt file, gives amount of data for which the distribution
is above the entropy threshold  
./test_entropy threshold softmax_output_file
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);

  if (argc < 2) {
    LOG(ERROR) << "Usage: split input_file";
    return 1;
  }
  
  float threshold = atof(argv[1]); 

  // Read softmax file
  int count = 0, total = 0;
  string str;
  std::ifstream file(argv[2]);
  vector<string> parts;
  while (std::getline(file, str)) {
    boost::split(parts, str, boost::is_any_of(" "));
    vector<float> prob;

    for (int i = 2; i < parts.size(); ++i) {
      prob.push_back(atof(parts[i].c_str()));
    }
    float entropy = get_entropy(prob);
    if (entropy >= threshold) {
      ++count;
    }
    ++total;
  }
  
  LOG(INFO) << "Item over entropy threshold: " << count;
  LOG(INFO) << "Percentage of items over entropy threshold: " << float(count)/total;

  return 0;
}

float get_entropy(vector<float> &prob) {
    float entropy = 0.0;
    for (int i = 0; i < prob.size(); i++) {
      entropy += (-prob[i] * ((log(prob[i])))/(log(2.0)));
    } 
    return entropy;
}
