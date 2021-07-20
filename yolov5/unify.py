import cv2
import numpy as np
import sophon.sail as sail
import torch
from .utils import non_max_suppression


class Unify(object):
    def __init__(self):
        # load bmodel
        self.sail_net = sail.Engine("../data/models/yolov5s.bmodel", 0, sail.IOMode.SYSIO)
        self.graph_name = self.sail_net.get_graph_names()[0]
        self.input_name = self.sail_net.get_input_names(self.graph_name)[0]
        # load pytorch model
        self.torch_model = torch.jit.load("../data/models/yolov5s.torchscript.pt")

        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.confThreshold = 0.5
        self.nmsThreshold = 0.5
        self.objThreshold = 0.5
        with open('../data/coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def preprocess(self, img, target_size=640):
        self.target_size = target_size
        h, w, c = img.shape
        # Calculate widht and height and paddings
        r_w = target_size / w
        r_h = target_size / h
        if r_h > r_w:
            tw = target_size
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((target_size - th) / 2)
            ty2 = target_size - th - ty1
        else:
            tw = int(r_h * w)
            th = target_size
            tx1 = int((target_size - tw) / 2)
            tx2 = target_size - tw - tx1
            ty1 = ty2 = 0
        # Resize long
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        # pad
        padded_img = cv2.copyMakeBorder(
            img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
        )
        # BGR => RGB
        resized_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        # to tensor
        image = resized_img.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, padded_img, (max(r_w, r_h), tx1, ty1)

    def func(self, outs):
        z = []  # inference output
        for i in range(self.nl):
            print(outs[i].shape)
            name = "_".join(list(map(str, outs[i].shape)))
            np.save("torch_{}.npy".format(name), outs[i])

            bs, _, ny, nx, nc, = outs[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if self.grid[i].shape[2:4] != outs[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-outs[i]))  ### sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

    def predict(self, tensor):
        # sail
        input_data = {self.input_name: np.array(tensor, dtype=np.float32)}
        output = self.sail_net.process(self.graph_name, input_data)
        sail_out = self.func(list(output.values()))
        # torch
        input_tensor = torch.from_numpy(tensor)
        th_out = self.torch_model(input_tensor)
        torch_out = self.func(th_out)
        return sail_out, torch_out

    def postprocess(self, pred, img, save_name, opt):
        pred = torch.tensor(pred)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = pred[0].numpy()
        for det in pred:
            left, top, right, bottom = list(map(int, det[:4]))
            score, id = det[4], int(det[-1])
            print(left, top, right, bottom, score, id)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), thickness=4)
        cv2.imwrite(save_name, img)
