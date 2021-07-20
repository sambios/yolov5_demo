import torch
import numpy as np
import cv2
import sys
import argparse

sys.path.append("../")
from yolov5.utils import non_max_suppression
from yolov5.pytorch import Detector


def main():
    parser = argparse.ArgumentParser(description="Demo of YOLOv5")
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    YOLOv5 = Detector()
    # srcimg = cv2.imread("data/images/bus.jpg")
    src_img = cv2.imread("../data/images/zidane.jpg")
    #src_img = cv2.imread("/home/yuan/tmp/testData/2020110117362000000022.jpg")

    img, padded_img, (ratio, tx1, ty1) = YOLOv5.preprocess(src_img, target_size=640)
    print("img.shape: {}".format(img.shape))
    #######################
    # dump input tensor
    np.save("torch_input.npy", img)
    dets = YOLOv5.predict(img)
    # print(dets.shape)
    # # dump output tensor
    # np.save("torch_output.npy", dets)
    # plot_img = YOLOv5.postprocess(padded_img, dets)
    # print(plot_img.shape)
    pred = torch.tensor(dets)
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    pred = pred[0].numpy()
    for det in pred:
        left, top, right, bottom = list(map(int, det[:4]))
        score, id = det[4], int(det[-1])
        print(left, top, right, bottom, score, id)
        cv2.rectangle(padded_img, (left, top), (right, bottom), (0, 0, 255), thickness=4)
    # cv2.imwrite("ccc.jpg", padded_img)

    cv2.imwrite("torch.jpg", padded_img)


if __name__ == '__main__':
    main()
