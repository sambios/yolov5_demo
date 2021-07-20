import argparse
import sys

import cv2
import numpy as np
import torch

sys.path.append("../")
from yolov5.utils import non_max_suppression
from yolov5.sail import Detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of YOLOv5")
    parser.add_argument('--bmodel', default="../data/models/yolov5s.bmodel", required=False)
    parser.add_argument('--input', default="../data/images/zidane.jpg", required=False)
    parser.add_argument('--tpu_id', default=0, type=int, required=False)
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()

    yolov5s = Detector(bmodel_path=opt.bmodel, tpu_id=opt.tpu_id)
    src_img = cv2.imread(opt.input)
    img, padded_img, (ratio, tx1, ty1) = yolov5s.preprocess(src_img, target_size=640)
    print("img.shape: {}".format(img.shape))
    #####################
    np.save("sail_input.npy", img)
    dets = yolov5s.predict(img)
    print(dets.shape)
    # dump output tensor
    np.save("sail_output.npy", dets)
    # plot_img = yolov5s.postprocess(padded_img, dets)
    # print(plot_img.shape)
    # cv2.imwrite("sail.jpg", plot_img)
    pred = torch.tensor(dets)
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    pred = pred[0].numpy()
    for det in pred:
        left, top, right, bottom = list(map(int, det[:4]))
        score, id = det[4], int(det[-1])
        print(left, top, right, bottom, score, id)
        cv2.rectangle(padded_img, (left, top), (right, bottom), (0, 0, 255), thickness=4)
    cv2.imwrite("sail.jpg", padded_img)