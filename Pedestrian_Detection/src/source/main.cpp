#include "hog.hpp"
#include <iostream>
#include <log.hpp>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <vector>

using namespace std;

#define DEFAULT_IMG_PATH                                                       \
  ("/home/guo/moDisk/gCode/Computer-Vision/Pedestrian_Detection/imgs/"         \
   "COCO_test2014_000000000014.jpg")

cv::Mat img;
cv::Rect2i patch_rect;
int click_flag = 0;

int init();
void process();
int release();

int init() {
  // do nothing
  return 0;
}

static void onMouse(int event, int x, int y, int flags, void *) {
  if (cv::EVENT_MOUSEMOVE == event) {
    if (click_flag) {
      // cv::rectangle(img, cv::Point2i(patch_rect.x, patch_rect.y),
      //               cv::Point2i(x, y), cv::Scalar(0, 255, 0), 1);
    }
  }

  if (cv::EVENT_LBUTTONDOWN == event) {
    if (0 == click_flag) {
      patch_rect.x = x;
      patch_rect.y = y;
      cv::circle(img, cv::Point2i(x, y), 3, cv::Scalar(255, 0, 0), -1);
      click_flag = 1;
    }
  }

  if (cv::EVENT_LBUTTONUP == event) {
    if (click_flag) {
      patch_rect.width = abs(x - patch_rect.x);
      patch_rect.height = abs(y - patch_rect.y);
      cv::rectangle(img, patch_rect, cv::Scalar(255, 0, 0), 2);
      click_flag = 0;
    }
  }
  cv::imshow("input_image", img);
}

void get_image_patch(cv::Mat &img_patch) {
  string sImagePath;
  cout << "Please input test image absolute path" << endl;
  getline(std::cin, sImagePath);

  if (sImagePath.empty()) {
    sImagePath = DEFAULT_IMG_PATH;
  }

  struct stat buf;
  int res = stat(sImagePath.c_str(), &buf);
  if (res != 0) {
    cout << "img path not valid" << endl;
    return;
  }

  cv::Mat ori_img = cv::imread(sImagePath, cv::IMREAD_GRAYSCALE);
  ori_img.copyTo(img);

  while (1) {
    cv::namedWindow("input_image", cv::WINDOW_NORMAL);
    cv::setMouseCallback("input_image", onMouse, 0);

    int key = cv::waitKey();
    if (key == 27 || key == 'q') {
      break;
    }
  }
  cv::destroyAllWindows();
  img_patch = cv::Mat(ori_img, patch_rect);
}

void process() {
  cv::Mat img_patch;
  get_image_patch(img_patch);
  extractHogFeature(img_patch);
}

int release() {
  // do nothing
  return 0;
}

int main() {
  init();
  process();
  release();

  return 0;
}