#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include <float.h>
#include <vector>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
using std::vector;

const int INTERVAL = 25;

/*
Read text file listing negatives image names and write two files:
1. text files with names of negative images
2. binary file with corresponding extracted features

./extract_features model_prototxt_file weights_of_network input_txt_file output_prefix num_iterations gpu
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
 
  LOG(INFO) << "starting...";
 
  string list_filename = argv[3];
  string features_filename_prefix = argv[4];
  int num_iterations = atoi(argv[5]);
  int gpu = atoi(argv[6]);  

  Caffe::SetDevice(gpu);
  Caffe::set_mode(Caffe::GPU);

  //get the net
  Net<float> caffe_test_net(argv[1]);
  
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  
  std::ifstream input_file(list_filename.c_str());

  std::ofstream output_list_file;
  std::ofstream output_feature_file;
  string output_list_filename = features_filename_prefix + ".txt";
  string output_features_filename = features_filename_prefix  + ".b";
  output_list_file.open(output_list_filename.c_str(), std::ios::app);
  output_feature_file.open(output_features_filename.c_str(), std::ios::app | std::ios::binary);

  string str;
  vector<string> parts;

  float loss;
  vector<Blob<float>* > bottom_vec;
  
  for (int i = 0; i < num_iterations; ++i) {
    LOG(INFO) << "Batch " << i; 
    
    // Run ForwardPrefilled 
    const vector<Blob<float>*>& result =
        caffe_test_net.Forward(bottom_vec, &loss);

    //LOG(INFO) << "size: " << result.size();
    //LOG(INFO) << "size1: " << result[0]->count();
    //LOG(INFO) << "size2: " << result[1]->count();
 
    int idx = 0; 
    const float* all_prob = result[1]->cpu_data();
    for (int j = 0; j < result[0]->count(); ++j) {
      float sum = 0;
      for (int k = 0; k < 205; ++k, ++idx) {
        output_feature_file.write((char *) &all_prob[idx], sizeof(float));    
        sum += all_prob[idx];
      }
      CHECK(sum <= 1.02);
      CHECK(sum >= 0.98);     
 
      std::getline(input_file, str);
      output_list_file << str << "\n";
      //LOG(INFO) << "True label: " << result[0]->cpu_data()[j];
    }   
    
    if (i % INTERVAL == 0) {
      LOG(INFO) << "Finished Batch  " << (i+1) << "/" << num_iterations;
    }
  }

  LOG(INFO) << "Finished.";
  output_feature_file.close();
  output_list_file.close();
  
  return 0;
}
