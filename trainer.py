from visual import Logger

import os
import os.path as osp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F

from data import VOCDetection, Augmentation, testTransform

from models.Loss import focalLoss, multiBoxLoss
from models.RetinaNet import retinaNet
from data.Anchor import Decoder
from eval import Testor

import argparse

parser = argparse.ArgumentParser(description = 'Training arguments for RetinaNet')
parser.add_argument('--batch_size', default = 20, type = int, help = "Number of inputs at once")
parser.add_argument('--lr', default = 1e-3, type = float, help = "initial learning rate")
parser.add_argument('--num_classes', default = 20, type = int, help = "Number of classes")
parser.add_argument('--cuda', default = True, type = bool, help = "Use cuda to train")
parser.add_argument('--experiment_name', default = 'VOC_default', type = str, help = "Name for logging")
parser.add_argument('--weight_decay', default = 1e-4, type = float, help = "weight decay rate")
parser.add_argument('--momentum', default = 0.9, type = float, help = "Momentum for SGD")
parser.add_argument('--means', default = None, nargs = '+', type = int, help = "Subtract means from samples")
parser.add_argument('--image_size', default = 300, type = int, help = "Image size to resize")
parser.add_argument('--max_iter', default = 2500000, type = int, help = "Maximum number of iteration")
parser.add_argument('--decay_steps', default = None, nargs = '+', type = int, help = "Decay the learning rate for each steps")
parser.add_argument('--optim', default = 'SGD', type = str, help = "Optimizer for training")
parser.add_argument('--loss_type', default = 'focal', type = str, help = "Loss function to use")
parser.add_argument('--resume', default = None, type = str, help = "Train again from paused ckpt")
args = parser.parse_args()

net = retinaNet(args.num_classes, 9, args.resume)

if args.cuda and torch.cuda.is_available():
    print(" [*] Set cuda: True")
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    print(" [*] Set cuda: False")
    #torch.set_default_tensor_type('torch.FloatTensor')

if args.resume:
    print(" [*] Train started from pretrained %s"%args.resume)
    net.load_state_dict(torch.load('./ckpt/%s'%args.resume))

logger = Logger('./visual/' + args.experiment_name)

opt = None
if args.optim.lower() == 'adam':
    opt = optim.Adam(net.parameters(), lr = args.lr)
else:
    opt = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
#opt = optim.Adam(net.parameters(), lr = args.lr)

criterion = None
if args.loss_type.lower() == 'ce':
    criterion = multiBoxLoss()
elif args.loss_type.lower() == 'focal':
    criterion = focalLoss()

print(" [*] Training is ready now!")

def train():
    net.train()
    net.module.freeze_bn()
    epoch = -1
    conf_loss = 0
    loc_loss = 0

    print(" [*] Loading dataset...")
    batch_iterator = None #?
    trainset = VOCDetection(os.getcwd() + "/data/VOC_root", [('2007', 'trainval'), 
            ('2012', 'trainval')], args.image_size, 
            Augmentation(args.image_size, args.means))
    train_loader = data.DataLoader(trainset, args.batch_size, num_workers = 4,
                shuffle = False, collate_fn = trainset.detection_collate, pin_memory = True)

    testset = VOCDetection(os.getcwd() + "/data/VOC_root", [('2007', 'test')],
                args.image_size, testTransform(args.image_size, args.means))
    test_loader = data.DataLoader(testset, args.batch_size, num_workers = 4,
                shuffle = False, collate_fn = testset.detection_collate, pin_memory = True)

    testor = Testor(testset, 'results/')
    decoder = Decoder(input_size = args.image_size, cuda = args.cuda)
    old_loss = 999.
    old_acc = 0.
    steps = 0 # decay step

    epoch_size = len(trainset) // args.batch_size

    start_iter = 0
    if args.resume:
        name = args.resume.split('.pth')[0]
        start_iter = int(name.split('_')[-1])
        epoch = int(start_iter / epoch_size)
        print(" [*] start itaration: %d"%start_iter)

    for iteration in range(start_iter, args.max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(train_loader)

        if iteration % epoch_size == 0:
            epoch += 1

        if iteration in args.decay_steps:
            steps += 1
            adjust_learning_rate(opt, 0.1, steps, args.lr)

        images, conf_targets, loc_targets = next(batch_iterator)
        images = Variable(images)
        conf_targets = Variable(conf_targets)
        loc_targets = Variable(loc_targets)

        if args.cuda:
            images = images.cuda()
            conf_targets = conf_targets.cuda()
            loc_targets = loc_targets.cuda()

        t0 = time.time()

        conf_preds, loc_preds = net(images)
        opt.zero_grad()

        c_loss, l_loss = criterion((conf_preds, loc_preds), (conf_targets, loc_targets))
        t2 = time.time()
        loss = (c_loss + l_loss)
        loss.backward()
        opt.step()
        t1 = time.time()

        logger.scalar_summary('c_loss', c_loss.data[0], iteration + 1)
        logger.scalar_summary('l_loss', l_loss.data[0], iteration + 1)
        logger.scalar_summary('loss', c_loss.data[0] + l_loss.data[0], iteration + 1)

        if (iteration % 10) == 0: # display period
            print(" [*] Epoch[%d], Iter %d || Loss: %.4f || c_loss: %.4f || l_loss: %.4f || Timer: %.4fsec"%(epoch, iteration, loss.data[0], c_loss.data[0], l_loss.data[0], (t1 - t0)))
        if (iteration % 5000) == 0: # evaluation period
            net.eval()

            test_loss = []
            total_acc = [] # TODO: get mAP
            all_boxes = [[[] for _ in range(len(testset))] for _ in range(args.num_classes + 1)]

            idx = 0
            for test_images, test_conf_targets, test_loc_targets in test_loader:
                test_images = Variable(test_images)
                test_conf_targets = Variable(test_conf_targets)
                test_loc_targets = Variable(test_loc_targets)

                if args.cuda:
                    test_images = test_images.cuda()
                    test_conf_targets = test_conf_targets.cuda()
                    test_loc_targets = test_loc_targets.cuda()

                test_conf_preds, test_loc_preds = net(test_images)
                c_loss, l_loss = criterion((test_conf_preds, test_loc_preds),
                                        (test_conf_targets, test_loc_targets))

                for i in range(len(test_images)):
                    conf, loc = test_conf_preds[i], test_loc_preds[i]
                    conf = F.softmax(conf, 1)
                    detections = decoder((conf.data, loc.data), conf_thresh = 0.01, nms_thresh = 0.45)
                    _, _, _, width, height = testset[idx]

                    for j in range(1, detections.size(1)):
                        dets = detections[0, j, :]
                        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                        dets = torch.masked_select(dets, mask).view(-1, 5)
                        if dets.dim() == 0:
                            continue
                        boxes = dets[:, 1:]
                        boxes[:, 0] *= width
                        boxes[:, 2] *= width
                        boxes[:, 1] *= height
                        boxes[:, 3] *= height
                        scores = dets[:, 0].cpu().numpy()
                        cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy = False)
                        all_boxes[j][idx] = cls_dets
                    idx += 1

                loss = c_loss.data[0] + l_loss.data[0]
                test_loss.append(loss)
            test_loss = np.mean(test_loss)
            test_acc = np.mean(total_acc) # ?
            print("  [*] Test loss: %.4f"%(test_loss))
            aps = testor.evaluate_detections(all_boxes, args.experiment_name, iteration)
            test_map = np.mean(aps)
            logger.scalar_summary('test_loss', test_loss, iteration + 1)
            logger.scalar_summary('test_mAP', test_map, iteration + 1)

            if test_loss < old_loss or (iteration % 10000) == 0:
                print("  [*] Save ckpt, iter: %d ar ckpt/"%iteration)
                file_path = 'ckpt/retina_%s_%d.pth'%(args.experiment_name, iteration)

                torch.save(net.state_dict(), file_path)
                if test_loss < old_loss:
                    old_loss = test_loss
            
            net.train() # back to train mode
            net.module.freeze_bn()

def adjust_learning_rate(optimizer, gamma, steps, _lr):
    lr = _lr * (gamma ** (steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    train()











