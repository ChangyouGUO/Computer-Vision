#include <hog.hpp>
#include <iostream>
#include <log.hpp>

using namespace std;

void extractHogFeature(cv::Mat img_patch) {
  // step1: preprocessing
  cv::resize(img_patch, img_patch, cv::Size(PATCH_WIDTH, PATCH_HEIGHT));
  cv::GaussianBlur(img_patch, img_patch, cv::Size(3, 3), 0.0);

#if (HOG_DEBUG_MODE)
  cout << "img_path" << endl;
  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      cout << (int)img_patch.at<uchar>(i, j) << " ";
    }
    cout << endl;
  }
  cout << "========" << endl;
#endif

  cv::Mat gx, gy;
  cv::Sobel(img_patch, gx, CV_16S, 1, 0, 1);
  cv::Sobel(img_patch, gy, CV_16S, 0, 1, 1);

#if (HOG_DEBUG_MODE)
  cout << "gx, gy" << endl;
  cv::Mat gx_32f, gy_32f;
  gx.convertTo(gx_32f, CV_32F);
  gy.convertTo(gy_32f, CV_32F);

  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      cout << gx_32f.at<float>(i, j) << " ";
    }
    cout << endl;
  }
  cout << endl;

  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      cout << gy_32f.at<float>(i, j) << " ";
    }
    cout << endl;
  }
  cout << "========" << endl;
#endif

  gx.convertTo(gx, CV_32F);
  gy.convertTo(gy, CV_32F);
  cv::Mat mag, angle;
  cv::cartToPolar(gx, gy, mag, angle, 1);

#if (HOG_DEBUG_MODE)
  cout << "mag, angle" << endl;
  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      cout << mag.at<float>(i, j) << " ";
    }
    cout << endl;
  }

  cout << endl;

  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      cout << angle.at<float>(i, j) << " ";
    }
    cout << endl;
  }
  cout << "========" << endl;

  cv::Mat gx_8U, gy_8U, mag_8U;
  gx.convertTo(gx_8U, CV_8U);
  cv::namedWindow("patch_image", cv::WINDOW_NORMAL);
  cv::imshow("patch_image", gx_8U);
  cv::waitKey(0);

  gy.convertTo(gy_8U, CV_8U);
  cv::namedWindow("patch_image", cv::WINDOW_NORMAL);
  cv::imshow("patch_image", gy_8U);
  cv::waitKey(0);

  mag.convertTo(mag_8U, CV_8U);
  cv::namedWindow("patch_image", cv::WINDOW_NORMAL);
  cv::imshow("patch_image", mag_8U);
  cv::waitKey(0);
#endif

  // step3: Calculate Histogram of Gradients in 8x8 cells
  int patch_col_cells = PATCH_WIDTH / CELL_SIZE;
  int patch_row_cells = PATCH_HEIGHT / CELL_SIZE;

  vector<float> histogram_bins(9 * patch_col_cells * patch_row_cells, 0);
  for (int row = 0; row < patch_row_cells; row++) {
    for (int col = 0; col < patch_col_cells; col++) {
      cv::Mat cell_mag = mag(
          cv::Rect2i(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE));
      cv::Mat cell_angle = angle(
          cv::Rect2i(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE));

      // per cell
      for (int j = 0; j < CELL_SIZE; j++) {
        for (int i = 0; i < CELL_SIZE; i++) {
          int point_angle = int(cell_angle.at<float>(j, i));
          float point_mag = cell_mag.at<float>(j, i);

          point_angle = point_angle % 180;

          float ratio = (point_angle % 20) / 20.0;

          int lower_index = point_angle / 20;
          int high_index = (lower_index + 1) % 9;

          histogram_bins[9 * (row * patch_col_cells + col) + lower_index] +=
              (1 - ratio) * point_mag;
          histogram_bins[9 * (row * patch_col_cells + col) + high_index] +=
              ratio * point_mag;

#if (HOG_DEBUG_MODE)
          if (row < 1 && col < 1) {
            cout << lower_index << "/" << high_index << "/" << ratio << " ";
          }
#endif
        }

#if (HOG_DEBUG_MODE)
        if (row < 1 && col < 1) {
          cout << endl;
        }
#endif
      }
    }
  }

#if (HOG_DEBUG_MODE)
  cout << "first cell historgram" << endl;
  for (int i = 0; i < 9; i++) {
    cout << histogram_bins[i] << " ";
  }
  cout << endl;
#endif

  // step4: 16x16 block normalization
  vector<vector<float>> normalized_histogram(
      patch_row_cells - 1, vector<float>(36 * (patch_col_cells - 1)));
  for (int row = 0; row < patch_row_cells - 1; row++) {
    for (int col = 0; col < patch_col_cells - 1; col++) {
      // get origin vector
      float len = 0;
      for (int i = 0; i < 36; i++) {
        if (i < 18) {
          normalized_histogram[row][col + i] =
              histogram_bins[row * (9 * patch_col_cells) + 9 * col + i];
        } else {
          normalized_histogram[row][col + i] =
              histogram_bins[(row + 1) * (9 * patch_col_cells) + 9 * col + i];
        }

        len += normalized_histogram[row][col + i] *
               normalized_histogram[row][col + i];
      }

      len = sqrt(len);

      // normalize
      for (int i = 0; i < 36; i++) {
        normalized_histogram[row][col + i] /= len;
      }
    }
  }

  // step5: visualize hog
  drawHogFeature(img_patch, histogram_bins);
}

void drawHogFeature(cv::Mat img_patch, vector<float> histogram_bins) {
  int patch_col_cells = PATCH_WIDTH / CELL_SIZE;
  int patch_row_cells = PATCH_HEIGHT / CELL_SIZE;

  cv::Mat hog_vis = cv::Mat(img_patch.rows * 2, img_patch.cols * 2, CV_8UC1,
                            cv::Scalar(255, 255, 255));
  vector<vector<vector<float>>> hog_data(
      patch_row_cells,
      vector<vector<float>>(patch_col_cells, vector<float>(9, 0)));

  int len = 0;
  for (size_t i = 0; i < histogram_bins.size(); i++) {
    int r = i / (9 * patch_col_cells);
    int c = (i % (9 * patch_col_cells)) / 9;

    len += histogram_bins[i] * histogram_bins[i];
    if (i % 9 == 8) {
      len = sqrt(len);
      for (size_t j = i - 8; j <= i; j++) {
        hog_data[r][c][j % 9] = histogram_bins[j] / len;
      }
      len = 0;
    }
  }

  // draw
  float delta = 20;
  for (size_t r = 0; r < hog_data.size(); r++) {
    for (size_t c = 0; c < hog_data[0].size(); c++) {
      for (int i = 0; i < 9; i++) {
        int cx = c * CELL_SIZE * 2 + CELL_SIZE;
        int cy = r * CELL_SIZE * 2 + CELL_SIZE;

        float radius = (CELL_SIZE - 1) * hog_data[r][c][i];

        float theta = (i * delta) * CV_PI / 180.0;
        float x_offset = radius * cos(theta);
        float y_offset = radius * sin(theta);

        cv::Point p1(cx + x_offset, cy - y_offset);
        cv::Point p2(cx - x_offset, cy + y_offset);

        cv::line(hog_vis, p1, p2, cv::Scalar(0, 0, 255), 1);
      }
    }
  }

  cv::namedWindow("hog", cv::WINDOW_NORMAL);
  cv::imshow("hog", hog_vis);
  int key = cv::waitKey(0);
  while (key != 'q') {
    cv::imshow("hog", hog_vis);
    key = cv::waitKey(0);
  }
}