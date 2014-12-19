#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  //top_k_ = this->layer_param_.accuracy_param().top_k();
  top_k_ = 1;
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

  int actual_num = 0;
  double threshold = 0.5;
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);

  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  vector<Blob<Dtype>*> vals;
  vals.push_back(bottom[0]);
  vector<Blob<Dtype>*> prob;
  Blob<Dtype>* blob(new Blob<Dtype>());
  prob.push_back(blob);
  layer.SetUp(vals, &prob);
  layer.Forward(vals, &prob);

  // Open file for writing threshold info 
  std::ofstream file;
  std::ostringstream s;
  s << "/home/sachinr/exp-data/week2/thres_" << threshold << ".txt";
  file.open(s.str().c_str(), std::ios::app);

  const Dtype* prob_data = prob[0]->cpu_data();
  for (int i = 0; i < num; ++i) {
    // Calculate entrpy  
    double entropy = 0.0;
    for (int j = 0; j < dim; ++j) {
      entropy += -(prob_data[i * dim + j] * ((log(prob_data[i * dim + j])))/(log(2.0)));
    } 
    //LOG(INFO) << "Entropy: " << entropy;
    //LOG(INFO) << "----------------------";
    if (entropy < threshold) {
      ++actual_num;

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
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == static_cast<int>(bottom_label[i])) {
          ++accuracy;
          break;
        }
      }
    }
  }
  LOG(INFO) << "Num Correct: " << accuracy;
  LOG(INFO) << "Num Clasified: " << actual_num;
  if (actual_num > 0) {
    (*top)[0]->mutable_cpu_data()[0] = accuracy / actual_num;
  }
  else {
    (*top)[0]->mutable_cpu_data()[0] = 1.0;
  }
  file << actual_num << "\n";
  file.close();
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyLayer);

}  // namespace caffe
