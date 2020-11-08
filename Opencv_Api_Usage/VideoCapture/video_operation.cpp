#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

vector<string> YOLO_CLASSES = {"__background__",
                               "person",
                               "bicycle",
                               "car",
                               "motorcycle",
                               "airplane",
                               "bus",
                               "train",
                               "truck",
                               "boat",
                               "traffic light",
                               "fire hydrant",
                               "stop sign",
                               "parking meter",
                               "bench",
                               "bird",
                               "cat",
                               "dog",
                               "horse",
                               "sheep",
                               "cow",
                               "elephant",
                               "bear",
                               "zebra",
                               "giraffe",
                               "backpack",
                               "umbrella",
                               "handbag",
                               "tie",
                               "suitcase",
                               "frisbee",
                               "skis",
                               "snowboard",
                               "sports ball",
                               "kite",
                               "baseball bat",
                               "baseball glove",
                               "skateboard",
                               "surfboard",
                               "tennis racket",
                               "bottle",
                               "wine glass",
                               "cup",
                               "fork",
                               "knife",
                               "spoon",
                               "bowl",
                               "banana",
                               "apple",
                               "sandwich",
                               "orange",
                               "broccoli",
                               "carrot",
                               "hot dog",
                               "pizza",
                               "donut",
                               "cake",
                               "chair",
                               "couch",
                               "potted plant",
                               "bed",
                               "dining table",
                               "toilet",
                               "tv",
                               "laptop",
                               "mouse",
                               "remote",
                               "keyboard",
                               "cell phone",
                               "microwave",
                               "oven",
                               "toaster",
                               "sink",
                               "refrigerator",
                               "book",
                               "clock",
                               "vase",
                               "scissors",
                               "teddy bear",
                               "hair drier",
                               "toothbrush"};

void printHelp() {
  cout << "usage:" << endl;
  cout << "./video_operation v video.avi" << endl;
  cout << "./video_operation c /dev/0" << endl;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printHelp();
  }
  string mode = argv[1];

  cout << YOLO_CLASSES[0] << " " << YOLO_CLASSES[1] << endl;

  if (mode == "v") {
    string video_file(argv[2]);
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened()) {
      cout << "Failed to open video: " << video_file << endl;
    }

    cv::Mat img;
    while (true) {
      bool success = cap.read(img);
      if (!success) {
        cout << "finish" << endl;
        break;
      }
      cv::imshow("video", img);
      if (cv::waitKey(100) == 'q') {
        break;
      }
    }

  } else if (mode == "c") {
  } else {
    printHelp();
  }
}
