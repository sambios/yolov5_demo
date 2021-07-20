import argparse
import sys

import cv2
import numpy as np

sys.path.append("../")
# from yolov5.utils import non_max_suppression
from yolov5.opencv import yolov5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='../data/images/zidane.jpg', help="image path")
    parser.add_argument('--net_type', default='yolov5s', choices=['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x'])
    args = parser.parse_args()

    yolonet = yolov5(args.net_type)
    src_img = cv2.imread(args.imgpath)
    img, padded_img, (ratio, tx1, ty1) = yolonet.preprocess(src_img, target_size=640)
    print("img.shape: {}".format(img.shape))

    np.save("cv_input.npy", img)
    dets = yolonet.detect(img)
    print(dets.shape)
    # dump output tensor
    np.save("cv_output.npy", dets)

    plot_img = yolonet.postprocess(padded_img, dets)

    print(plot_img.shape)
    cv2.imwrite("cv.jpg", plot_img)
