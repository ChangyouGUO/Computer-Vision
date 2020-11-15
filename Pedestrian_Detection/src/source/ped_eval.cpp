#include "ped_eval.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <LibSVM/svm.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>  // HOGDescriptor

using namespace std;
using namespace cv;
using namespace ml;

#define KERNEL (SVM::RBF)

static char *fileline = NULL;
static int max_line_len;

static char *readline(FILE *input) {
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

void read_param(vector<double> &mean, vector<double> &std) {
  char paramPath[100];
  sprintf(paramPath, "../model/perclosParam");
  FILE *fp = fopen(paramPath, "r");

  if (fp == NULL) {
    printf("can't open input file %s\n", paramPath);
    exit(1);
  }

  max_line_len = 1024;
  fileline = (char *)malloc(max_line_len * sizeof(char));
  int j = 0;
  while (readline(fp) != NULL && j < 2) {
    char *value, *endptr;
    size_t i = 0;

    if (j == 0) {
      if (i == 0) {
        value = strtok(fileline, " \t\n");
        mean[i] = strtod(value, &endptr);
      }
      for (i = 1; i < mean.size(); i++) {
        value = strtok(NULL, " \t\n");
        mean[i] = strtod(value, &endptr);
      }
      j++;
    } else {
      if (i == 0) {
        value = strtok(fileline, " \t\n");
        std[i] = strtod(value, &endptr);
      }
      for (i = 1; i < mean.size(); i++) {
        value = strtok(NULL, " \t\n");
        std[i] = strtod(value, &endptr);
      }
      j++;
    }
  }
  fclose(fp);
}

int main(int argc, char **argv) {
  clock_t start, finish;
  double totaltime = 0.0;

  // load model
  svm_model *model;
  char modelPath[100];
  sprintf(modelPath, "../model/perclos.model");

  if ((model = svm_load_model(modelPath)) == 0) {
    fprintf(stderr, "can't open model file %s\n", modelPath);
    exit(1);
  }

#if (PREDICT_PROBABILITY)
  if (svm_check_probability_model(model) == 0) {
    printf("Model does not support probabiliy estimates\n");
    exit(2);
  } else {
    printf(
        "Model supports probability estimates, but disabled in prediction.\n");
  }
#endif

  // load mean and std
  vector<double> mean(324, 0);
  vector<double> std(324, 1);
  read_param(mean, std);

  // Image file
  cv::String pedPath = "../data/ped*.jpg";
  cv::String naturePath = "../data/nature*.jpg";
  vector<cv::String> pedFn;
  vector<cv::String> natureFn;

  cv::glob(pedPath, pedFn, false);
  cv::glob(naturePath, natureFn, false);

  cout << "Total image num:" << pedFn.size() + natureFn.size() << endl;

  // hog
  cv::HOGDescriptor hog(
      Size(IMAGE_SIZE, IMAGE_SIZE),
      Size(PIXEL_PER_CELL * CELL_PER_BLOCK, PIXEL_PER_CELL * CELL_PER_BLOCK),
      Size(PIXEL_PER_CELL, PIXEL_PER_CELL),
      Size(PIXEL_PER_CELL, PIXEL_PER_CELL), ORIENT, 1, (-1.0),
      HOGDescriptor::L2Hys, 0.2, true);
  int DescriptorDim;

  // svm node
  struct svm_node *x;
  double predict_label;
  int correct = 0;
  int total = 0;

#ifdef PREDICT_PROBABILITY
  int svm_type = svm_get_svm_type(model);
  int nr_class = svm_get_nr_class(model);
  double *prob_estimates = NULL;
  int j;

  if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
    printf(
        "Prob. model for test data: target value = predicted value + z,\nz: "
        "Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",
        svm_get_svr_probability(model));
  else {
    int *labels = (int *)malloc(nr_class * sizeof(int));
    svm_get_labels(model, labels);
    prob_estimates = (double *)malloc(nr_class * sizeof(double));
    printf("labels");
    for (j = 0; j < nr_class; j++) printf(" %d", labels[j]);
    printf("\n");
    free(labels);
  }
#endif

  // ped Label = 1
  for (size_t i = 0; i < pedFn.size(); i++) {
    Mat image = imread(pedFn[i]);

    start = clock();
    vector<float> descriptors;
    hog.compute(image, descriptors);

    if (i == 0)  // initial
    {
      DescriptorDim = descriptors.size();
      // trainFeatureMat = Mat::zeros(pedFn.size()+natureFn.size(),
      // DescriptorDim, CV_32FC1);  //Data format trainLabelMat =
      // Mat::zeros(pedFn.size()+natureFn.size(), 1, CV_32SC1);

      x = (struct svm_node *)malloc(
          (DescriptorDim + 1) *
          sizeof(struct svm_node));  // featue node, one more for index = -1
    }

    int j = 0;
    for (j = 0; j < DescriptorDim; j++) {  // each feature
      // trainFeatureMat.at<float>(i, j) = descriptors[j];
      x[j].index = (int)j + 1;
      x[j].value = (double)(descriptors[j] - mean[j]) / (std[j] + 1e-5);
    }
    x[j].index = -1;

#ifdef PREDICT_PROBABILITY
    if (svm_type == C_SVC || svm_type == NU_SVC) {
      predict_label = svm_predict_probability(model, x, prob_estimates);
      printf("%g", predict_label);
      for (j = 0; j < nr_class; j++) printf(" %g", prob_estimates[j]);
      printf("\n");
    } else {
      predict_label = svm_predict(model, x);
      printf("%.17g\n", predict_label);
    }
#endif

    if ((int)predict_label == 1) {
      ++correct;
    }
    ++total;

    finish = clock();
    totaltime += finish - start;
  }

  // nature Label = 0
  for (size_t i = 0; i < natureFn.size(); i++) {
    Mat image = imread(natureFn[i]);

    start = clock();
    vector<float> descriptors;
    hog.compute(image, descriptors);

    int j = 0;
    for (j = 0; j < DescriptorDim; j++) {
      // trainFeatureMat.at<float>(i+pedFn.size(), j) = descriptors[j];
      x[j].index = (int)j + 1;
      x[j].value = (double)(descriptors[j] - mean[j]) / (std[j] + 1e-5);
    }
    x[j].index = -1;

#ifdef PREDICT_PROBABILITY
    if (svm_type == C_SVC || svm_type == NU_SVC) {
      predict_label = svm_predict_probability(model, x, prob_estimates);
      printf("%g", predict_label);
      for (j = 0; j < nr_class; j++) printf(" %g", prob_estimates[j]);
      printf("\n");
    } else {
      predict_label = svm_predict(model, x);
      printf("%.17g\n", predict_label);
    }
#endif

    if ((int)predict_label == 0) {
      ++correct;
    }
    ++total;

    finish = clock();
    totaltime += finish - start;
  }
  printf("Accuracy = %g%% (%d/%d)\n", (double)correct / total * 100, correct,
         total);

  totaltime = (double)(totaltime) / 1000;
  cout << "cost time: " << endl;
  cout << totaltime / (natureFn.size() + pedFn.size()) << "ms/per image"
       << endl;

  svm_free_and_destroy_model(&model);
  free(x);
  free(fileline);
#ifdef PREDICT_PROBABILITY
  free(prob_estimates);
#endif
  return 0;
}