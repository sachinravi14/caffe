#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <dirent.h>

#include <iomanip>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/size.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;
using std::string;
using std::vector;

vector<string> listFile(string path);

string NFS_PATH = "/n/fs/lsun/imageTurk/";
string IMAGES_DIR = "/home/sachinr/code/images-mount-space/";

string NEG_IMAGES_DIR = "/home/sachinr/code/neg-images-mount-space";
string NEG_PREFIX = "neg_category_";

string LOC = "/data/sachinr/places-data/";
string FILENAME = "all_negatives.txt";

/*
Write list of negatives to text file 
./write_negatives
*/
int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);

  string input_prefix = argv[2];

  std::ofstream file;
  string filename = LOC + FILENAME;
  file.open(filename.c_str(), std::ios::app);

  vector<string> files;
  files = listFile(NEG_IMAGES_DIR);
  for (int i = 0; i < files.size(); ++i) {
    string filename = NEG_IMAGES_DIR + "/" + files[i];
    LOG(INFO) << filename;
    std::ifstream neg_file(filename.c_str());
    string str;

    vector<string> parts;
    while (std::getline(neg_file, str)) {
      boost::split(parts, str, boost::is_any_of(" "));
      parts[0].replace(0, NFS_PATH.size(), IMAGES_DIR);

      file << parts[0] << " " << parts[1] << "\n";
    }
    LOG(INFO) << "Finished " << (i+1) << " files";
  } 

  file.close();

  return 0;
}

vector<string> listFile(string path){
  DIR *pDIR;
  vector<string> files;
  struct dirent *entry;

  if( (pDIR=opendir(path.c_str())) ){
    while(entry = readdir(pDIR)) {
      if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ) {
        string str_name = string(entry->d_name);
        if (str_name.find(NEG_PREFIX) != std::string::npos) {
          files.push_back(str_name);
        }
      }
    }
    closedir(pDIR);
  }

  return files;
}
