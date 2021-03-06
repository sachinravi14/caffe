#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
//#include <omp>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "time.h"
#include <omp.h>

namespace caffe {

int getFileSize(const std::string &filepath) {
    int filesize = -1;

    struct stat fileStats;
    if(stat(filepath.c_str(), &fileStats) != -1)
      filesize = fileStats.st_size;

    return filesize;
}

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  LOG(ERROR) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(ReadImageToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                         new_height, new_width, &datum));
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
  LOG(ERROR) << "Data layer initialized!";
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  vector<Datum> vdatum(1000, Datum());
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();

  // datum scales
  //clock_t loadtime = 0;
  //clock_t trantime = 0;
  const int lines_size = lines_.size();

  
  LOG(INFO) << "read index start";  
  vector<int> current_id;
  current_id.assign(batch_size, 0);
  for (int item_id = 0; item_id < batch_size; ++item_id)
  {
    CHECK_GT(lines_size, lines_id_);
    while (getFileSize(lines_[lines_id_].first.c_str()) <= 0) { // Skip images that have size 0
      LOG(INFO) << "Skipping: " << lines_[lines_id_].first;
      lines_id_++;
    }
    current_id[item_id] = lines_id_;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  //omp_set_dynamic(0);     // Explicitly disable dynamic teams
  //omp_set_num_threads(4); // Use 4 threads for all consecutive parallel regions

  //for (int i = 0; i<batch_size; i++)
  //{
  //  LOG(ERROR) << current_id[i];
  //}

  LOG(INFO) << "read data start";  
  #pragma omp parallel num_threads(15)
  {
    #pragma omp for
    for (int item_id = 0; item_id < batch_size; ++item_id)
    {
      int id = current_id[item_id];
      // LOG(ERROR) << item_id << ":" << id << ":" << lines_[id].first;
      if (!ReadImageToDatum(lines_[id].first,
            lines_[id].second,
            new_height, new_width, &(vdatum[item_id]))) {
        LOG(ERROR) << id << ": "<< lines_[id].first << " (ERRORLOAD)";
      }
      this->data_transformer_.Transform(item_id, vdatum[item_id], this->mean_, top_data);
      top_label[item_id] = vdatum[item_id].label();
    }
  //  LOG(ERROR) << "threads="<<omp_get_num_threads()<<endl;
  }
  LOG(INFO) << "read data finish";  
  

  /*
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    //clock_t t1 = clock();
    CHECK_GT(lines_size, lines_id_);
    if (!ReadImageToDatum(lines_[lines_id_].first,
          lines_[lines_id_].second,
          new_height, new_width, &datum)) {
    	LOG(ERROR) << lines_id_ << ": "<< lines_[lines_id_].first;
    	lines_id_++;
      continue;
    }
    //loadtime += clock() - t1;

    
    // Apply transformations (mirror, crop...) to the data
    //t1 = clock();
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);
    //trantime += clock() - t1;

    top_label[item_id] = datum.label();


    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  */
  
  //printf("Loadtime: %d; Trantime: %d\n", loadtime, trantime);
}

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
