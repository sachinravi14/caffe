#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//extern DataProcess ExternData;
namespace caffe {

const string PATH = "/home/sachinr/";
const string DIST_FILE_PREFIX = "network_dist_";
const string CONF_MATRIX_FILE_PREFIX = "confusion_matrix";
const string FILE_FORMAT = ".txt";

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

    return buf;
}

template <typename Dtype>
void write(string name, int id, int dim, int label, const Dtype* dist) {
  std::ofstream file;
  file.open(name.c_str(), std::ios::app);
 
  //string key((char *)TestData.Key[TestData.MiniBatchDataID[id]].mv_data);
  //key.erase(std::remove(key.begin(), key.end(), '\n'), key.end());
  //key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
  //key.erase( std::remove_if(key.begin(), key.end(), ::isspace ), key.end() );
  //file << key << " "; 
  //file << label << " ";
  
  for (int i = 0; i < dim; ++i) {
    file << dist[dim * id + i];
    if (i < dim - 1) { 
      file << " "; 
    }
    else { 
      file << "\n"; 
    }   
  }
  file.close();
}

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);

  Dtype posvalidnum = 0;
	Dtype negvalidnum = 0;
	Dtype posaccuracy = 0;
	Dtype negaccuracy = 0;

  // Get softmax values
  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  vector<Blob<Dtype>*> vals;
  vector<Blob<Dtype>*> prob;
  Blob<Dtype>* blob(new Blob<Dtype>());
  const Dtype* prob_data; 

  std::ostringstream pos_file_name; 
  std::ostringstream neg_file_name;  
  std::ostringstream conf_matrix_file_name; 
  std::ofstream conf_matrix_file;

  // Compute Softmax Values 
  vals.push_back(bottom[0]);
  prob.push_back(blob);
  layer.SetUp(vals, &prob);
  layer.Forward(vals, &prob);
  prob_data = prob[0]->cpu_data(); 

  // For writing softmax output
  pos_file_name << PATH << DIST_FILE_PREFIX << "pos_" << currentDateTime() << FILE_FORMAT;
  neg_file_name << PATH << DIST_FILE_PREFIX << "neg_" << currentDateTime() << FILE_FORMAT;
  
  // For confusion matrix
  conf_matrix_file_name << PATH << CONF_MATRIX_FILE_PREFIX << FILE_FORMAT;
  conf_matrix_file.open(conf_matrix_file_name.str().c_str(), std::ios::app);

  int count_none = 0;
  //top_k_ = 2;
  for (int i = 0; i < num; ++i) {
		int ori = static_cast<int>(bottom_label[i]);
    bool isneg = ori<-0.5;
	
		if ( !isneg ) {
      // Write probability distribution for positive data 
      write(pos_file_name.str(), i, dim, ori, prob_data); 

			++posvalidnum;
		}	
		else {
      // Write probability distribution for negative data
      write(neg_file_name.str(), i, dim, -ori-1, prob_data);	

      ++negvalidnum;
		}
		
		// Top-k accuracy
		std::vector<std::pair<Dtype, int> > bottom_data_vector;
		for (int j = 0; j < dim; ++j) {
			bottom_data_vector.push_back(
				std::make_pair(bottom_data[i * dim + j], j));
		}
		std::partial_sort(
		  bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
			bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
		
    // check if true label is in top k predictions
		for (int k = 0; k < 1; k++) { 
      if (!isneg) {
        conf_matrix_file << ori << " " << bottom_data_vector[k].second << "\n"; 
        if (bottom_data_vector[k].second == ori) {
		  	  ++posaccuracy;
	  		  break;
  			} 
			}
      else {
        if (bottom_data_vector[k].second !=  -ori-1) {
          ++negaccuracy;
          break;
        } 
      }
		}

  }
  conf_matrix_file.close();

  if (negvalidnum == 0) {
    accuracy = ((float) posaccuracy)/posvalidnum;
  } else if (posvalidnum == 0) {
    accuracy = ((float) negaccuracy)/negvalidnum;
  } else if (negvalidnum > 0) {	
	  accuracy = 0.5*( posaccuracy / ((float)posvalidnum + 0.0001)) + 0.5*(negaccuracy / ((float)negvalidnum + 0.0001));
  }
	
  (*top)[0]->mutable_cpu_data()[0] = accuracy;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyLayer);

}  // namespace caffe
