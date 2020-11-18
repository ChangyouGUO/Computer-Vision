#include "ped_predict.hpp"

#include <getopt.h>
#include <sys/stat.h>
#include <iostream>
#include <log.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace ml;

#define DEFAULT_IMG_PATH                                               \
  ("/home/guo/moDisk/gCode/Computer-Vision/Pedestrian_Detection/imgs/" \
   "COCO_test2014_000000000014.jpg")

static char *fileline = NULL;
static int max_line_len = 1024;

char param_path[BUFFER_LEN];
char model_path[BUFFER_LEN];
PedestrianClassifier *pedestrian_classifier;

void arg_parse(int argc, char **argv);
int init();
void process();
int release();

static char *ReadLine(FILE *input) {
  int len;

  if (fgets(fileline, max_line_len, input) == NULL) return NULL;

  while (strrchr(fileline, '\n') == NULL) {
    max_line_len *= 2;
    fileline = (char *)realloc(fileline, max_line_len);
    len = (int)strlen(fileline);
    if (fgets(fileline + len, max_line_len - len, input) == NULL) break;
  }
  return fileline;
}

void set_params() {
  FILE *fp = fopen(param_path, "r");

  if (fp == NULL) {
    printf("can't open input file %s\n", param_path);
    exit(1);
  }

  fileline = (char *)malloc(max_line_len * sizeof(char));
  int j = 0;
  while (ReadLine(fp) != NULL && j < 2) {
    char *value, *endptr;
    size_t i = 0;

    if (j == 0) {
      if (i == 0) {
        value = strtok(fileline, " \t\n");
        pedestrian_classifier->mean_[i] = strtod(value, &endptr);
      }
      for (i = 1; i < pedestrian_classifier->mean_.size(); i++) {
        value = strtok(NULL, " \t\n");
        pedestrian_classifier->mean_[i] = strtod(value, &endptr);
      }
      j++;
    } else {
      if (i == 0) {
        value = strtok(fileline, " \t\n");
        pedestrian_classifier->std_[i] = strtod(value, &endptr);
      }
      for (i = 1; i < pedestrian_classifier->mean_.size(); i++) {
        value = strtok(NULL, " \t\n");
        pedestrian_classifier->std_[i] = strtod(value, &endptr);
      }
      j++;
    }
  }
  fclose(fp);
}

cv::Mat img;
cv::Rect2i patch_rect;
int click_flag = 0;

int init() {
  pedestrian_classifier = new PedestrianClassifier();

  // load model
  if ((pedestrian_classifier->model_ = svm_load_model(model_path)) == 0) {
    fprintf(stderr, "can't open model file %s\n", model_path);
    exit(1);
  }

//  check model
#if (PREDICT_PROBABILITY)
  if (svm_check_probability_model(pedestrian_classifier->model_) == 0) {
    printf("Model does not support probabiliy estimates\n");
    exit(2);
  } else {
    printf(
        "Model supports probability estimates, but disabled in prediction.\n");
  }
#endif

  // set params
  set_params();

// init svm_type_, nr_class_, prob_estimate
#if (PREDICT_PROBABILITY)
  pedestrian_classifier->svm_type_ =
      svm_get_svm_type(pedestrian_classifier->model_);

  pedestrian_classifier->nr_class_ =
      svm_get_nr_class(pedestrian_classifier->model_);

  if (pedestrian_classifier->svm_type_ == NU_SVR ||
      pedestrian_classifier->svm_type_ == EPSILON_SVR) {
    printf(
        "Prob. model for test data: target value = predicted value + z,\nz: "
        "Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",
        svm_get_svr_probability(pedestrian_classifier->model_));
  } else {
    int labels[pedestrian_classifier->nr_class_];
    svm_get_labels(pedestrian_classifier->model_, labels);
    pedestrian_classifier->prob_estimate_ =
        (double *)malloc(pedestrian_classifier->nr_class_ * sizeof(double));
    printf("labels");
    for (int j = 0; j < pedestrian_classifier->nr_class_; j++)
      printf(" %d", labels[j]);
    printf("\n");
  }
#endif

  // init x, featue node, one more for index = -1
  pedestrian_classifier->x_ = (struct svm_node *)malloc(
      (pedestrian_classifier->descriptor_dim_ + 1) * sizeof(struct svm_node));

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
  cv::resize(img_patch, img_patch, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

  vector<float> patch_descriptors;
  pedestrian_classifier->hog_descriptor_.compute(img_patch, patch_descriptors);

  int j = 0;
  for (j = 0; j < pedestrian_classifier->descriptor_dim_; j++) {
    pedestrian_classifier->x_[j].index = (int)j + 1;
    pedestrian_classifier->x_[j].value =
        (double)(patch_descriptors[j] - pedestrian_classifier->mean_[j]) /
        (pedestrian_classifier->std_[j] + 1e-5);
  }
  pedestrian_classifier->x_[j].index = -1;

  double predict_label;
#if (PREDICT_PROBABILITY)
  if (pedestrian_classifier->svm_type_ == C_SVC ||
      pedestrian_classifier->svm_type_ == NU_SVC) {
    predict_label = svm_predict_probability(
        pedestrian_classifier->model_, pedestrian_classifier->x_,
        pedestrian_classifier->prob_estimate_);
    printf("%g", predict_label);
    for (int j = 0; j < pedestrian_classifier->nr_class_; j++)
      printf(" %g", pedestrian_classifier->prob_estimate_[j]);
    printf("\n");
  }
#else
  predict_label =
      svm_predict(pedestrian_classifier->model_, pedestrian_classifier->x_);
  printf("%.17g\n", predict_label);
#endif
}

int release() {
  svm_free_and_destroy_model(&(pedestrian_classifier->model_));
  free(pedestrian_classifier->x_);
  free(fileline);
#ifdef PREDICT_PROBABILITY
  free(pedestrian_classifier->prob_estimate_);
#endif
  free(pedestrian_classifier);
  return 0;
}

int main(int argc, char **argv) {
  arg_parse(argc, argv);
  init();
  process();
  release();

  return 0;
}

void arg_parse(int argc, char **argv) {
  int opt;
  char *arg_str = "p:m:";

  while ((opt = getopt(argc, argv, arg_str)) != -1) {
    switch (opt) {
      case 'p':
        snprintf(param_path, BUFFER_LEN, "%s", optarg);
        printf("-p: %s\n", param_path);
        break;
      case 'm':
        snprintf(model_path, BUFFER_LEN, "%s", optarg);
        printf("-m: %s\n", model_path);
        break;
      default:
        break;
    }
  }
}