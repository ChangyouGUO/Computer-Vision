#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import argparse
import glob
import random

image_width = 64
image_height = 128
negative_num_per_image = 3


def main():
    parser = argparse.ArgumentParser(description='prepare INRIAPerson dataset')
    parser.add_argument('--pos_path',
                        type=str,
                        help="input positive image path")
    parser.add_argument('--neg_path',
                        type=str,
                        help="input negative image path")
    parser.add_argument('--output_path', type=str, help="output image path")
    opt = parser.parse_args()
    print(opt)

    pos_images = glob.glob(opt.pos_path + "/*")
    neg_images = glob.glob(opt.neg_path + "/*")

    output_pos_path = opt.output_path + "/pos/"
    output_neg_path = opt.output_path + "/neg/"

    if (not os.path.exists(output_pos_path)):
        os.mkdir(output_pos_path)
    if (not os.path.exists(output_neg_path)):
        os.mkdir(output_neg_path)

    # pos
    for file in pos_images:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if (len(image) > image_height and len(image[0]) > image_width):
            image = image[16:16 + 128, 16:16 + 64]

        image_save_path = output_pos_path + file.split("/")[-1]
        cv2.imwrite(image_save_path, image)

    # neg
    for file in neg_images:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        for i in range(negative_num_per_image):
            x = int(random.random() * (len(image[0]) - 64))
            y = int(random.random() * (len(image) - 128))

            patch_image = image[y:y + 128, x:x + 64]
            image_name = file.split("/")[-1]
            image_save_path = output_neg_path + image_name.split(
                '.')[0] + "_" + str(i) + "." + image_name.split('.')[1]
            cv2.imwrite(image_save_path, patch_image)


if __name__ == "__main__":
    main()
