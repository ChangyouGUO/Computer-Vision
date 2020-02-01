from skimage.feature import hog
import numpy as np
import cv2

def main():
    face = np.array(cv2.imread("./face/000007.jpg", 0))
    print(face.shape)
    hogFeature, hogImage = hog(face, visualise=True)

    while(cv2.waitKey(0) != 'q'):
        cv2.imshow("face", hogImage)

if __name__ == '__main__':
    main()