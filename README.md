# Realtime Multi-object Detection Pipeline

_Note: this repo is currently under heavy development. Not ready for general consumption. So, please refrain from using it in production._

The goal of this project is to buid a real-time state-of-the-art multi-object detection pipeline that can be jointly train in single stage.

The PyTorch implementation will Weave together these techniques and methods from various research papers:

* Single Shot MultiBox Detector (SSD)
* YOLOv3 real-time properties
* Focal loss for dense object detection (RetinaNet)
* Non Maximum Suppression (NMS)
* Scalable object detection using deep neural networks
* Faster R-CNN tricks

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
