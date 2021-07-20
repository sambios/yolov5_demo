## Introduction

Demo of YOLOv5.

## Dependency

- python>=3.6
- opencv-python

### sail engine

- BMNNSDK2-1684 >=2.2.0

### pytorch==1.6.0

```bash
# CUDA 10.2
pip install torch==1.6.0 torchvision==0.7.0

# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Demo

cd to `demo_yolov5/yolov5`

### sail as backend

```bash
python sail.py
```

### opencv as backend

```bash
python opencv.py
```

### pytorch as backend

```bash
python pytorch.py
```

### compare the result of sail to pytorch

after run the demo of sail and pytorch,

cd path to `tests/`, run `compare.py`

```bash
$ python sail.py
bmcpu init: skip cpu_user_defined
open usercpu.so, init user_cpu_init
[BMRT][load_bmodel:723] INFO : Loading bmodel from [../data/models/yolov5s.bmodel]. Thanks for your patience...
[BMRT][load_bmodel:705] INFO : pre net num: 0, load net num: 1
img.shape: (1, 3, 640, 640)
(1, 25200, 85)
(640, 640, 3)
$ python pytorch.py
img.shape: (1, 3, 640, 640)
(1, 25200, 85)
(640, 640, 3)
$ ls
opencv.py  __pycache__  pytorch.py  sail_input.npy  sail.jpg  sail_output.npy  sail.py  torch_input.npy  torch.jpg  torch_output.npy  utils.py
$ cd ../tests/
$ python compare.py
True
False
```

### compare the result of opencv to pytorch

```bash
$ python opencv.py
img.shape: (1, 3, 640, 640)
(1, 3, 80, 80, 85)
(1, 3, 40, 40, 85)
(1, 3, 20, 20, 85)
(1, 25200, 85)
(640, 640, 3)

$ python compare.py
True
False
False
False
False
```
