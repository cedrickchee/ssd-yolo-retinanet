import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim = True)) + x_max

class multiBoxLoss(nn.Module):
    def __init__(self):
        super(multiBoxLoss, self).__init__()

    def forward(self, predictions, targets):
        """
            Args:
                predictions (tuple): (conf_preds, loc_preds)
                    conf_preds shape: [batch, n_anchors, num_cls]
                    loc_preds shape: [batch, n_anchors, 4]
                targets (tensor): (conf_targets, loc_targets)
                    conf_targets shape: [batch, n_anchors]
                    loc_targets shape: [batch, n_anchors, 4]
        """

        conf_preds, loc_preds = predictions
        conf_targets, loc_targets = targets

        ############### Confiden Loss part ###############
        num = loc_preds.size(0)

        # cross entropy loss
        ignored = conf_targets < 0
        conf_targets[ignored] = 0
        pos = conf_targets > 0
        batch_conf = conf_preds.view(-1, conf_preds.size(2))

        gathered_conf = batch_conf.gather(1, conf_targets.view(-1, 1))
        loss_c = log_sum_exp(batch_conf) - gathered_conf

        # Hard negative mining
        loss_c[pos.view(-1)] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending = True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim = True)

        # 3:1 in paper
        num_neg = torch.clamp(3 * num_pos, max = pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence loss including pos and neg examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_preds)
        neg_idx = neg.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[(pos_idx + neg_idx).gt(0)].view(-1, conf_preds.size(2))
        targets_weighted = conf_targets[(pos + neg).gt(0)]
        conf_loss = F.cross_entropy(conf_p, targets_weighted, size_average = False)

        ############# Localization Loss part ##############
        pos = conf_targets > 0 # ignore background
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        loc_p = loc_preds[pos_idx].view(-1, 4)
        loc_t = loc_targets[pos_idx].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_p, loc_t, size_average = False)

        num_pos = pos.long().sum(1, keepdim = True)
        N = max(num_pos.data.sum(), 1) # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes 
        conf_loss /= N # exclude number of background?
        loc_loss /= N

        return conf_loss, loc_loss

class focalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 0.25):
        """
            focusing is parameter that can adjust the rate at which easy
            examples are down-weighted.
            alpha may be set by inverse class frequency or treated as a hyper-param

            If you don't want to balance factor, set alpha to 1
            If you don't want to focusing factor, set gamma to 1 
            which is same as normal cross entropy loss
        """
        super(focalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets):
        """
            Args:
                predictions (tuple): (conf_preds, loc_preds)
                    conf_preds shape: [batch, n_anchors, num_cls]
                    loc_preds shape: [batch, n_anchors, 4]
                targets (tensor): (conf_targets, loc_targets)
                    conf_targets shape: [batch, n_anchors]
                    loc_targets shape: [batch, n_anchors, 4]
        """

        conf_preds, loc_preds = predictions
        conf_targets, loc_targets = targets

        ############### Confiden Loss part ###############
        """
        #focal loss implementation(1)
        pos_cls = conf_targets > -1 # exclude ignored anchors
        mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
        conf_t = conf_targets[pos_cls].view(-1).clone()

        p = F.softmax(conf_p, 1)
        p = p.clamp(1e-7, 1. - 1e-7) # to avoid loss going to inf
        c_mask = conf_p.data.new(conf_p.size(0), conf_p.size(1)).fill_(0)
        c_mask = Variable(c_mask)
        ids = conf_t.view(-1, 1)
        c_mask.scatter_(1, ids, 1.)

        p_t = (p*c_mask).sum(1).view(-1, 1)
        p_t_log = p_t.log()
        # This is focal loss presented in ther paper eq(5)
        conf_loss = -self.alpha * ((1 - p_t)**self.gamma * p_t_log)
        conf_loss = conf_loss.sum()
        """
    
        # focal loss implementation(2)
        pos_cls = conf_targets >-1
        mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
        p_t_log = -F.cross_entropy(conf_p, conf_targets[pos_cls], size_average = False)
        p_t = torch.exp(p_t_log)

        # This is focal loss presented in the paper eq(5)
        conf_loss = -self.alpha * ((1 - p_t)**self.gamma * p_t_log)

        ############# Localization Loss part ##############
        pos = conf_targets > 0 # ignore background
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        loc_p = loc_preds[pos_idx].view(-1, 4)
        loc_t = loc_targets[pos_idx].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_p, loc_t, size_average = False)

        num_pos = pos.long().sum(1, keepdim = True)
        N = max(num_pos.data.sum(), 1) # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes 
        conf_loss /= N # exclude number of background?
        loc_loss /= N

        return conf_loss, loc_loss

    def one_hot(self, x, n):
        y = torch.eye(n)
        return y[x]


if __name__ == '__main__':

    focal_loss = focalLoss()
    f_loss = focal_loss(x, y)
    print(" [*] f_loss:", f_loss)
