# RetinaNet-pytorch
This repo aims to implement a "Focal loss for Dense Object Detection"

### Requirements
- python 3
- [Pytorch 0.4](https://github.com/pytorch/pytorch#from-source)
- [torchvision](https://github.com/pytorch/vision)
- numpy
- [tensorflow](https://www.tensorflow.org/install/)

### Training
```Shell
# Select the script that you want to train for reproducing a results
./retina_ce_sgd_0.001.sh
# For the focal loss use ./retina_focal_sgd_0.0001.sh
```

You can see the details in trainer.py

### VOC Dataset
#####Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```
#####Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### TODO

- [x] Building RetinaNet
- [x] Apply cross entropy loss and focal loss(still checking it's works or not)
- [ ] Comparing btw CE loss and focal loss
- [ ] Report results on VOC
- [ ] Report results on COCO
- [ ] Use relative path for easy reproducing of result

### References

- [Keras version](https://github.com/fizyr/keras-retinanet)
    - Easy to read and very active repository. This repo drive me complete my code
- [Another pytorch version](https://github.com/kuangliu/pytorch-retinanet)
    - Almost all code are followed from this repository, but i am not sure this repo works

#### Author
jtlee at omnious dot com

Feel free to contact me about code.
