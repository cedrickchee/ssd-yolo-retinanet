import os
from data.voc0712 import VOC_CLASSES as labelmap

import pickle
import xml.etree.ElementTree as ET
import numpy as np

keep_difficult = False
class Testor(object):
    def __init__(self, dataset, save_folder, YEAR = 2007):
        self.dataset = dataset
        self.save_folder = save_folder # root_path
        self.YEAR = YEAR

    def xml2rec(self, filename):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not keep_difficult and difficult:
                continue
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
            objects.append(obj_struct)
        return objects

    def load_label(self, anno_path, testset_file):
        with open(testset_file, 'r') as f:
            lines = f.readlines()
        image_names = [x.strip() for x in lines]

        if not os.path.exists(anno_path + '/annots.pkl'):
            recs = {}
            for i, image_name in enumerate(image_names):
                recs[image_name] = self.xml2rec(anno_path + '/%s.xml'%image_name)
            print("  [*] Saving cached annots to %s"%(anno_path + '/annots.pkl'))
            with open(anno_path + '/annots.pkl', 'wb') as f:
                pickle.dump(recs, f)
        else:
            print("  [*] load cached annots")
            with open(anno_path + '/annots.pkl', 'rb') as f:
                recs = pickle.load(f)
        return image_names, recs

    def write_results_file(self, all_boxes, exp_name, iteration):
        save_path = os.path.join(self.save_folder, "%s_%s"%(exp_name, iteration))
        if not os.path.exists(save_path):
            print("  [!] No save folder exists, mkdir %s"%save_path)
            os.makedirs(save_path)
        else:
            print("  [*] Save folder found, result will be written at %s"%save_path)

        for cls_idx, cls_name in enumerate(labelmap):
            file_name = 'det_test_%s.txt'%cls_name
            file_path = os.path.join(save_path, file_name)
            with open(file_path, 'wt') as f:
                for im_idx, idx in enumerate(self.dataset.ids):
                    dets = all_boxes[cls_idx+1][im_idx]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(idx[1], dets[k, -1],
                                    dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

    def read_result(self, cls_name, exp_name, iteration):
        det_filename = os.path.join(self.save_folder, "%s_%s"%(exp_name, iteration), 'det_test_%s.txt'%cls_name)
        with open(det_filename, 'r') as f:
            lines = f.readlines()
        return lines

    def get_ap(self, recall, precision, at11 = False):
        if at11:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])

                ap = ap + p / 11.
        else:
            mrec = np.concatenate(([0.], recall, [1.]))
            mprec = np.concatenate(([0.], precision, [1.]))

            for i in range(mprec.size - 1, 0, -1):
                mprec[i-1] = np.maximum(mprec[i-1], mprec[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i+1] - mrec[i]) * mprec[i+1])
        return ap

    def class_eval(self, image_names, recs, cls_name, exp_name, iteration, ovthresh = 0.5):
        class_recs = {}
        npos = 0
        for image_name in image_names:
            R = [obj for obj in recs[image_name] if obj['name'] == cls_name]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
            npos = npos + len(R)
            class_recs[image_name] = {'bbox': bbox, 'det': det}

        lines = self.read_result(cls_name, exp_name, iteration)

        if any(lines) == 1:
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                            (BBGT[:, 2] - BBGT[:, 0]) *
                            (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                if ovmax > ovthresh:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)

            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.get_ap(rec, prec, True)
        else:
            rec = -1.
            prec = -1.
            ap = -1.
        return rec, prec, ap

    def evaluate_detections(self, box_list, exp_name, iteration):
        self.write_results_file(box_list, exp_name, iteration)

        image_names, recs = self.load_label('data/VOC_root/VOC%s/Annotations'%self.YEAR, 'data/VOC_root/VOC%s/ImageSets/Main/test.txt'%self.YEAR)

        aps = []
        for i, cls_name in enumerate(labelmap):
            rec, prec, ap = self.class_eval(image_names, recs, cls_name, exp_name, iteration)
            aps += [ap]
            print(" [*] AP for {} = {:.4f}".format(cls_name, ap))
            with open(os.path.join(self.save_folder, "%s_%s"%(exp_name, iteration), cls_name + 'perf.pkl'), 'wb') as f:
                pickle.dump({'recall': rec, 'precision': prec, 'ap': ap}, f)

        print(" [*] Mean AP = {:.4f}".format(np.mean(aps)))
        print("Results:")
        for i, ap in enumerate(aps):
            print(" [*] Class {} ap: {:.3f}".format(labelmap[i], ap))

        return aps

