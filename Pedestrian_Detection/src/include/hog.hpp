#ifndef _SRC_INCLUDE_HOG_
#define _SRC_INCLUDE_HOG_

#include <opencv2/opencv.hpp>
#include <vector>

#define CELL_SIZE (8)
#define PATCH_HEIGHT (128)
#define PATCH_WIDTH (64)

void extractHogFeature(cv::Mat img_patch);

void drawHogFeature(cv::Mat img_path, std::vector<float> histogram_bins);

#endif /* _SRC_INCLUDE_HOG_ */