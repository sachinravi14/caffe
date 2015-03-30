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

double threshold = 4.9;

const float UPPER_BOUND = 0.70;
const float LOWER_BOUND = 0.07;
const int TOP_K = 5;

const int INTERVAL = 25;

string FILENAME = "/data/sachinr/places-data/all_negatives.txt";
string USEFUL_FILENAME = "/data/sachinr/places-data/useful_negatives.txt";
string USELESS_FILENAME = "/data/sachinr/places-data/useless_negatives.txt";

float get_entropy(vector<float> prob);

/*
./write_useful_negatives file_prefix
From file prefix, reads in image list file and binary file with feature probability info for
each image and selects images to write to lmdb based on criteria. 
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
  double total = 0, added = 0; 
  while (std::getline(image_list_file, str)) {
    ++total;
    vector<float> prob;
    vector<std::pair<float, int> > bottom_data_vector;
    for (int i = 0; i < 205; ++i) {
      float p = 0;
      feature_file.read((char *)&p, sizeof(float));
      prob.push_back(p);
      bottom_data_vector.push_back(std::make_pair(p, i));
    
      if (total == 1 || total == 2 || total == 1001 || total == 1002) {
        LOG(INFO) << "Prob: " << i << " " << p;
      }
    }
    LOG(INFO) << "--------------";  
    
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
      //file << str << "\n"; 
    } 
  }   
     
  return 0;
}

float get_entropy(vector<float> prob) {
  float entropy = 0.0;
  for (int j = 0; j < prob.size(); ++j) {
    entropy -= prob[j] * (log(std::max(prob[j], float(FLT_MIN)))/(log(2.0)));
  }

  return entropy;
}
