import cv2

def main():
    cap = cv2.VideoCapture("/home/guo/myDisk/Dataset/metoak/mo_adas_video_202006131110.avi")
    if not cap.isOpened():
        print("cannot open!!!")
    while(True):
        ret, frame = cap.read()
        cv2.imshow("video", frame)
        cv2.waitKey(100)


if __name__ == "__main__":
    main()