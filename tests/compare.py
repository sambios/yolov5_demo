import numpy as np


def compare(path_a, path_b):
    a = np.load(path_a)
    b = np.load(path_b)
    ret = np.allclose(a, b)
    print(ret)
    return ret


def sail_vs_torch():
    compare(path_a="../yolov5/sail_input.npy",
            path_b="../yolov5/torch_input.npy")

    compare(path_a="../yolov5/sail_output.npy",
            path_b="../yolov5/torch_output.npy")

    compare(path_a="../yolov5/sail_1_3_20_20_85.npy",
            path_b="../yolov5/torch_1_3_20_20_85.npy")

    compare(path_a="../yolov5/sail_1_3_40_40_85.npy",
            path_b="../yolov5/torch_1_3_40_40_85.npy")

    compare(path_a="../yolov5/sail_1_3_80_80_85.npy",
            path_b="../yolov5/torch_1_3_80_80_85.npy")


def cv_vs_torch():
    compare(path_a="../yolov5/cv_input.npy",
            path_b="../yolov5/torch_input.npy")

    compare(path_a="../yolov5/cv_output.npy",
            path_b="../yolov5/torch_output.npy")

    compare(path_a="../yolov5/cv_1_3_20_20_85.npy",
            path_b="../yolov5/torch_1_3_20_20_85.npy")

    compare(path_a="../yolov5/cv_1_3_40_40_85.npy",
            path_b="../yolov5/torch_1_3_40_40_85.npy")

    compare(path_a="../yolov5/cv_1_3_80_80_85.npy",
            path_b="../yolov5/torch_1_3_80_80_85.npy")


if __name__ == '__main__':
    # sail_vs_torch()
    cv_vs_torch()
