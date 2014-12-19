#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/size.hpp"

using caffe::Datum;

int size(MDB_cursor* mdb_cursor, MDB_val mdb_key, MDB_val mdb_value) {
  Datum datum;
  int count = 0;
  do {
    // just a dummy operation
    datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
    //const string& data = datum.data();
    /*int size_in_datum = std::max<int>(datum.data().size(),
      datum.float_data_size());*/
    count++;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files";
    }
  } while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT)
                  == MDB_SUCCESS);
  return count; 
}
