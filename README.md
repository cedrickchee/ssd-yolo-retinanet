# Realtime Multi-object Detection Pipeline

_Note: this repo is currently under heavy development. It's not ready for general consumption. So, please refrain yourself from using it in production._

The goal of this project is to buid a single end-to-end deep learning model for more accurate and faster (near real-time) multi-object detection that can be train in single-pass of multiple different pieces:

* Single Shot MultiBox Detector (SSD)
* YOLOv3 real-time properties
* Focal loss for dense object detection (RetinaNet)
* Non Maximum Suppression (NMS)
* Scalable object detection using deep neural networks
* Faster R-CNN tricks

These techniques and methods from various research papers will be implemented using PyTorch.

We will be using Pascal VOC2007 dataset.

### Requirements
- Python 3
- Pytorch 0.4
- numpy
- fastai PyTorch library

### Training

```shell
# Select the script that you want to train for reproducing a results
./retina_ce_sgd_0.001.sh
# For the focal loss use ./retina_focal_sgd_0.0001.sh
```

You can see the details in trainer.py

### VOC Dataset

##### Download VOC2007 trainval & test

```shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### TODO

- [x] Build SSD + YOLO model
- [x] Apply cross entropy loss and focal loss
- [x] Compare between CE loss and focal loss
- [ ] Report results on VOC
    - currently achieved 50mAP on VOC2007.
- [ ] Report results on COCO
- [ ] Use relative path for easy reproducing of result
