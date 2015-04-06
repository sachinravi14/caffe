#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/*
This accuracy layer should only used for test set which contains images
belonging to class not in training set
*/

template <typename Dtype>
float get_entropy(vector<Dtype> prob) {
  float entropy = 0.0;
  for (int j = 0; j < prob.size(); ++j) {
    entropy -= prob[j] * (log(std::max(prob[j], Dtype(FLT_MIN)))/(log(2.0)));
  }

  return entropy;
}

template <typename Dtype>
void NoneEntropyAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void NoneEntropyAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void NoneEntropyAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::vector<Dtype> bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(bottom_data[i * dim + j]);
    }
    // check if entropy of softmax is greater than cutoff
    if (get_entropy(bottom_data_vector) >= entropy_cutoff) {
      ++accuracy;
      break;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(NoneEntropyAccuracyLayer);

}  // namespace caffe
