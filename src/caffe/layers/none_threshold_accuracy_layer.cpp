#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

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
void NoneThresholdAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void NoneThresholdAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void NoneThresholdAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype wrong = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  
  // Get softmax values
  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  vector<Blob<Dtype>*> vals;
  vector<Blob<Dtype>*> prob;
  Blob<Dtype>* blob(new Blob<Dtype>());
  const Dtype* prob_data;

  vals.push_back(bottom[0]);
  prob.push_back(blob);
  layer.SetUp(vals, &prob);
  layer.Forward(vals, &prob);
  prob_data = prob[0]->cpu_data(); 
 
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::vector<std::pair<Dtype, int> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(
          std::make_pair(prob_data[i * dim + j], j));
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + 1,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    
    // check if top class has prob >= threshold
    if (bottom_data_vector[0].first > threshold) {
      LOG(INFO) << "Above: " << bottom_data_vector[0].first;
      ++wrong;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = (num - wrong)/num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(NoneThresholdAccuracyLayer);

}  // namespace caffe
