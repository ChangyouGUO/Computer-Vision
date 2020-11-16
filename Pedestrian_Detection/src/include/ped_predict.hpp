#ifndef _SRC_INCLUDE_PED_PREDICT_HPP_
#define _SRC_INCLUDE_PED_PREDICT_HPP_

#include <LibSVM/svm.h>
#include <opencv2/opencv.hpp>

#define IMAGE_WIDTH (64)
#define IMAGE_HEIGHT (128)
#define PIXEL_PER_CELL (8)
#define CELL_PER_BLOCK (2)
#define ORIENT (9)
#define BUFFER_LEN (256)
#define KERNEL (SVM::RBF)
#define PREDICT_PROBABILITY (1)

class PedestrianClassifier {
public:
  int descriptor_dim_;
  std::vector<double> mean_;
  std::vector<double> std_;

  cv::HOGDescriptor hog_descriptor_;

  struct svm_node *x_;
  int svm_type_;
  int nr_class_;
  double *prob_estimate_;
  svm_model *model_;

  PedestrianClassifier()
      : descriptor_dim_(ORIENT * CELL_PER_BLOCK * CELL_PER_BLOCK *
                        (IMAGE_WIDTH / PIXEL_PER_CELL - 1) *
                        (IMAGE_HEIGHT / PIXEL_PER_CELL - 1)),
        hog_descriptor_(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT),
                        cv::Size(PIXEL_PER_CELL * CELL_PER_BLOCK,
                                 PIXEL_PER_CELL * CELL_PER_BLOCK),
                        cv::Size(PIXEL_PER_CELL, PIXEL_PER_CELL),
                        cv::Size(PIXEL_PER_CELL, PIXEL_PER_CELL), ORIENT, 1,
                        (-1.0), cv::HOGDescriptor::L2Hys, 0.2, true),
        mean_(descriptor_dim_, 0), std_(descriptor_dim_, 1) {}
};

#endif //  _SRC_INCLUDE_PED_PREDICT_HPP_