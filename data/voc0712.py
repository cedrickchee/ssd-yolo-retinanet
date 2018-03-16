"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

This code is updated from https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from .Anchor import Encoder
    
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class_to_idx = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
keep_difficult = False

def xml2Tensor(target):
    size = target.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    res = []
    for obj in target.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if not keep_difficult and difficult:
            continue
        name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            # no need to -1, it's 0-based index
            cur_pt = int(bbox.find(pt).text)
            cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
            bndbox.append(cur_pt)
        label_idx = class_to_idx[name]
        bndbox.append(label_idx)
        res += [bndbox]
    return res

class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    
    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, input_size, augmentation = None, 
            dataset_name='VOC0712'):

        self.root = root
        self.image_set = image_sets
        self.input_size = input_size
        self.augment = augmentation
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        
        self.encoder = Encoder(input_size = input_size)
    
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        img, boxes, labels, h, w = self.pull_item(index)
        return img, boxes, labels, w, h
        
    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        
        target = np.array(xml2Tensor(target))
        boxes, labels = target[:, :4].astype(np.float32), target[:, 4].astype(np.int64)
            
        if self.augment is not None:
            img, boxes, labels = self.augment(img, boxes, labels)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(boxes), torch.from_numpy(labels), height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        target = xml2Tensor(anno)
        boxes, labels = target[:, :4], target[:, 4]
        return img_id[1], boxes, labels
    
    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form
        
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
            
        Argument:
            index (int): index of img to show
        Return:
            ensorized version of img, squeezed    
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def detection_collate(self, batch):
        
        imgs = []
        loc_targets = []
        conf_targets = []

        for sample in batch:
            imgs.append(sample[0])
            conf_target, loc_target = self.encoder(sample[1], sample[2])
            conf_targets.append(conf_target)
            loc_targets.append(loc_target)
        return torch.stack(imgs, 0), torch.stack(conf_targets), torch.stack(loc_targets)

if __name__ == '__main__':
    from .augmentation import Augmentation

    image_size = 300 # size to resize
    dataset = VOCDetection(os.getcwd() + "/data/VOC_root", [('2007', 'trainval'),
            ('2012', 'trainval')], image_size,
            Augmentation(image_size, (104, 123, 114)))
    dataloader = data.DataLoader(dataset, batch_size = 32,
            shuffle = True, collate_fn = dataset.detection_collate, num_workers = 1)

    for images, conf_targets, loc_targets in dataloader:
        print(" [*] images:", images.size())
        print(" [*] conf_targets:", conf_targets.size())
        print(" [*] loc_targets:", loc_targets.size())
        break


    testset = VOCDetection(os.getcwd() + "/data/VOC_root", [('2007', 'test')],
            image_size, Augmentation(image_size, (104, 123, 114)))
    testloader = data.DataLoader(testset, batch_size = 32,
            shuffle = False, collate_fn = testset.detection_collate, num_workers = 1)

    for images, conf_targets, loc_targets in testloader:
        print(" [*] test images:", images.size())
        print(" [*] test conf_targets:", conf_targets.size())
        print(" [*] test loc_targets:", loc_targets.size())
        break

