import sys

import cv2
import numpy as np
import sophon.sail as sail


# from .utils import non_max_suppression, sigmoid


# input: x.1, [1, 3, 640, 640], float32, scale: 1
# output: 147, [1, 3, 80, 80, 85], float32, scale: 1
# output: 148, [1, 3, 40, 40, 85], float32, scale: 1
# output: 149, [1, 3, 20, 20, 85], float32, scale: 1
class Detector(object):
    def __init__(self, bmodel_path, tpu_id):
        # load bmodel
        self.net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        # generate anchor
        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        # post-process threshold
        self.confThreshold = 0.5
        self.nmsThreshold = 0.5
        self.objThreshold = 0.1

        with open('../data/coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def predict(self, tensor):
        # blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
        input_data = {self.input_name: np.array(tensor, dtype=np.float32)}
        output = self.net.process(self.graph_name, input_data)
        z = []  # inference output

        for i, feat in enumerate(output.values()):
            print(feat.shape)
            name = "_".join(list(map(str, feat.shape)))
            np.save("sail_{}.npy".format(name), feat)
            bs, _, ny, nx, nc = feat.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if self.grid[i].shape[2:4] != feat.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-feat))  ### sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

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

    def postprocess(self, frame, outs):
        # frameHeight = frame.shape[0]
        # frameWidth = frame.shape[1]
        # ratioh, ratiow = frameHeight / 640, frameWidth / 640

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        # (25200, 85)
        for out in outs:
            # (85,)
            for detection in out:
                # class scores
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold and detection[4] > self.objThreshold:
                    # center_x = int(detection[0] * ratiow)
                    # center_y = int(detection[1] * ratioh)
                    # width = int(detection[2] * ratiow)
                    # height = int(detection[3] * ratioh)
                    center_x = int(detection[0])
                    center_y = int(detection[1])
                    width = int(detection[2])
                    height = int(detection[3])

                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence) * detection[4])
                    boxes.append([left, top, width, height])

        # Perform nms to eliminate redundant overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # print(indices.shape)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame
