import math
import numpy as np

import torch
from .utils import jaccard, nms, meshgrid

class Anchor(object):
    def __init__(self):
        self.anchor_areas   = [32*32., 64*64., 128*128., 256*256., 512*512.]
        self.strides        = [8, 16, 32, 64, 128]
        self.aspect_ratios  = [1/2., 1/1., 2/1.]
        self.scale_ratios   = [1., pow(2, 1/3.), pow(2, 2/3.)]
        self.variances      = [0.1, 0.2]
        self.anchor_wh      = self._get_anchor_wh()

    def _get_anchor_wh(self):
        anchor_wh = []
        for area in self.anchor_areas:
            for aspect in self.aspect_ratios:
                h = math.sqrt(area / aspect)
                w = aspect * h
                for scale in self.scale_ratios:
                    anchor_h = h * scale
                    anchor_w = w * scale
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i+3)).ceil() for i in range(num_fms)]
        # feature map size p_3 to p_7

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = self.strides[i]
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5
            xy = xy.view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2) / fm_size
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2) / input_size
            box = torch.cat([xy, wh], 3) # [x, y, w, h]
            boxes.append(box.view(-1, 4))
        boxes = torch.cat(boxes, 0)
        # clip is optional
        boxes.clamp_(max = 1, min = 0) # reference call
        return boxes

    def _change_box_order(self, boxes, order):
        options = ['xyxy2xywh', 'xywh2xyxy']
        assert order in options
        a = boxes[:, :2]
        b = boxes[:, 2:]
        if order == options[0]:
            return torch.cat([(a + b)/2, (b - a)], 1)
        else:
            return torch.cat([a - (b/2), a + (b/2)], 1)

class Encoder(Anchor):
    def __init__(self, input_size):
        super(Encoder, self).__init__()

        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size])
        elif isinstance(input_size, tuple):
            input_size = torch.Tensor(input_size)
        else:
            raise ValueError("input_size(%s) is not expected type"%str(type(input_size)))

        self.anchor_boxes = self._get_anchor_boxes(input_size)

    def __call__(self, boxes, labels):
        """
            Args
                boxes: bounding boxes of (xmin, ymin, xmax, ymax), [#obj, 4]
                labels: object class labels, [#obj,]
                input_size: model input size of (w, h), (int/tuple)
            returns
                conf_targets: encoded class labels, [#anchors,]
                loc_targets: encoded bounding boxes, [#anchors, 4]
        """
        point_form = torch.cat((self.anchor_boxes[:, :2] - self.anchor_boxes[:, 2:] / 2, self.anchor_boxes[:, :2] + self.anchor_boxes[:, 2:] / 2), 1)
        ious = jaccard(boxes, point_form)
        boxes = self._change_box_order(boxes, 'xyxy2xywh')

        best_anchor_ious, best_anchor_idx = ious.max(1, keepdim = True)
        best_boxes_ious, best_boxes_idx = ious.max(0, keepdim = True)
        
        best_anchor_ious.squeeze_(1)
        best_anchor_idx.squeeze_(1)
        best_boxes_ious.squeeze_(0)
        best_boxes_idx.squeeze_(0)
        best_boxes_ious.index_fill_(0, best_anchor_idx, 2)

        for j in range(best_anchor_idx.size(0)):
            best_boxes_idx[best_anchor_idx[j]] = j

        boxes = boxes[best_boxes_idx]
    
        loc_xy = (boxes[:, :2] - self.anchor_boxes[:, :2]) / (self.anchor_boxes[:, 2:] * self.variances[0])
        loc_wh = torch.log(boxes[:, 2:] / self.anchor_boxes[:, 2:]) / self.variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        conf_targets = 1 + labels[best_boxes_idx] # add background label as a 0

        conf_targets[best_boxes_ious < 0.5] = 0 # background
        ignore = (best_boxes_ious > 0.4) & (best_boxes_ious < 0.5)
        conf_targets[ignore] = -1

        return conf_targets, loc_targets

class Decoder(Anchor):
    def __init__(self, input_size, cuda = False):
        super(Decoder, self).__init__()

        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size])
        elif isinstance(input_size, tuple):
            input_size = torch.Tensor(input_size)
        else:
            raise ValueError("input_size(%s) is not expected type"%str(type(input_size)))

        self.anchor_boxes = self._get_anchor_boxes(input_size)
        if cuda:
            self.anchor_boxes = self.anchor_boxes.cuda()
    def __call__(self, predictions, conf_thresh = 0.5, nms_thresh = 0.5):
        """
            Args
                predictions: (tuple), (conf_preds, loc_preds)
                conf_preds: predicted class labels, [#anchors, #classes]
                loc_preds: predicted locations, [#anchors, 4]
                input_size: model input size of (w, h), (int/tuple)
            returns
                output: (scores, boxes) [#obj, 5]
                boxes: decode box locations, [#obj, 4]
                labels: class labels for each box [#obj,]
        """
        conf_preds, loc_preds = predictions

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * self.anchor_boxes[:, 2:] * self.variances[0] + self.anchor_boxes[:, :2]
        wh = (loc_wh * self.variances[1]).exp() * self.anchor_boxes[:, 2:]

        decoded_boxes = torch.cat([xy - wh/2, xy + wh/2], 1) # [#anchor, 4]
        # decoded_boxes
        decoded_boxes = torch.clamp(decoded_boxes, min = 0., max = 1.)
        
        top_k = 200 # temporal
        output = torch.zeros(1, conf_preds.size(1), top_k, 5)
        for cl in range(1, conf_preds.size(1)): # ignore background
            # For each class, perform nms
            c_mask = conf_preds[:, cl].gt(conf_thresh)
            scores = conf_preds[:, cl][c_mask]
            if scores.dim() == 0:
                continue

            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            ids, count = nms(boxes, scores, nms_thresh)
            output[0, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                        boxes[ids[:count]]), 1)
        flat = output.view(-1, 5)
        _, idx = flat[:, 0].sort(0)
        _, rank = idx.sort(0)
        flat[(rank >= top_k).unsqueeze(1).expand_as(flat)].fill_(0)

        return output

def one_hot(x, n):
    y = torch.eye(n)

    if x[0] < 0:
        x[0] = 0 

    t = y[x]
    return t

if __name__ == '__main__':
    import numpy as np
    from .augmentation import Augmentation

    augment = Augmentation(300, [107, 113, 123])

    
    encoder = Encoder(input_size = 300)
    imgs = np.zeros((500, 500, 3))
    real_boxes = np.array([[20, 30, 70, 80], [320, 23, 500, 120]], dtype = np.float32)
    boxes = []
    for box in real_boxes:
        bndbox = []
        for i, pt in enumerate(box):
            cur_pt = pt / 500 if i % 2 == 0 else pt / 500
            bndbox.append(cur_pt)
        boxes.append(bndbox)
    boxes = np.array(boxes, dtype = np.float32)

    labels = np.array([[1], [7]], dtype = np.int64)
    print("before augment", boxes)
    imgs, boxes, labels = augment(imgs, boxes, labels)
    print("after augment", boxes)
    print(labels)
    boxes, labels = torch.from_numpy(boxes), torch.from_numpy(labels)

    conf_target, loc_target = encoder(boxes, labels)

    print(" [*] Encoding results")
    print(" [*] conf_target:")
    print(conf_target)
    print(" [*] loc_target:")
    print(loc_target)


    decoder = Decoder(input_size = 300)

    print(" [*] Decoding result")
    conf_pred = np.stack([one_hot(target, 9).numpy() for target in conf_target])
    conf_pred = conf_pred.reshape(-1, 9)
    output = decoder((torch.FloatTensor(conf_pred), loc_target))

    all_boxes = [[[] for _ in range(1)] for _ in range(9)]
    for j in range(1, 9):
        dets = output[0, j, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.dim() == 0:
            continue
        boxes = dets[:, 1:]
        scores = dets[:, 0].cpu().numpy()
        cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy = False)
        all_boxes[j][0] = cls_dets

    #print(" [*] label:")
    #print(output[:, 0])
    print(" [*] box:")
    print(output)







