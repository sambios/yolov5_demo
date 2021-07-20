import numpy as np


def parse_input_ref_data():
    input_data = np.fromfile("./input_ref_data.dat.bmrt", dtype=np.float32)
    print(input_data.shape)
    input_data = np.reshape(input_data, newshape=(1, 3, 640, -1))
    print(input_data.shape)
    sail_x = np.load("./sail_input.npy")
    ret = np.allclose(sail_x, input_data)
    print(ret)


def parse_output_ref_data():
    output_data = np.fromfile("./output_ref_data.dat.bmrt", dtype=np.float32)
    print(output_data.shape)
    sail_a = np.load("./sail_1_3_80_80_85.npy").ravel()
    sail_b = np.load("./sail_1_3_40_40_85.npy").ravel()
    sail_c = np.load("./sail_1_3_20_20_85.npy").ravel()
    print(sail_a.shape)
    print(sail_b.shape)
    print(sail_c.shape)
    final = np.concatenate((sail_a, sail_b, sail_c), axis=-1)
    print(final.shape)
    ret = np.allclose(output_data, final)
    print(ret)


if __name__ == '__main__':
    parse_input_ref_data()
    parse_output_ref_data()
