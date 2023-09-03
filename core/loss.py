import torch
import torch.nn as nn
import numpy as np

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        # q:[B,2048]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['NA'], 1), 
            torch.mean(contrast_pairs['PA'], 1), 
            contrast_pairs['PB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['NB'], 1), 
            torch.mean(contrast_pairs['PB'], 1), 
            contrast_pairs['PA']
        )

        loss = HA_refinement + HB_refinement
        return loss
        
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.ce_criterion = nn.BCELoss()
        # 150 最好
        self.margin = 150

    def get_absloss(self,contrast_pairs):
        feat_act = contrast_pairs["PA"]
        feat_bkg = contrast_pairs['PB']
        
        loss_act = self.margin - \
                   torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um

    def get_tripletloss(self,label, contrast_pair):
        n = label.size(0) # get batch_size
        class_sim_idx =[]
        action_list = contrast_pair['PA']

        # 遍历20个类
        for i in range(label.size(1)):
            class_sim_idx.append(set())

        # get the same label
        # for i in range(n):
        #     for j in range(i+1,n):
        #         label0 = label[i].cpu().numpy()
        #         label1 = label[j].cpu().numpy()
        #         if (label0 == label1).all():
        #             l = label0.tolist()
        #             # idx = l.index(1)
        #             for idx in range(len(l)):
        #                 if l[idx]==1: 
        #                     class_sim_idx[idx].add(i)
        #                     class_sim_idx[idx].add(j)


        # 获得每个类别有那些视频
        for i in range(n):
            label0 = label[i].cpu().numpy()
            for j in range(len(label0)):
                if label0[j]==1:
                    class_sim_idx[j].add(i)

        triplet_loss = torch.FloatTensor([0.])
        
        # 计算距离
        distence = torch.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                action0 = torch.mean(action_list[i],dim=1)
                action1 = torch.mean(action_list[j],dim=1)
                d = 1- torch.sum(action0*action1,dim=0)/(torch.norm(action0, 2, dim=0) * torch.norm(action1, 2, dim=0)) 
                distence[i][j] = d
                distence[j][i] = d

        # 遍历 20 个类别
        for i in range(label.size(1)):
            if len(class_sim_idx[i])<=1:
                # 如果第i个类别为0
                continue
            for idx in class_sim_idx[i]:
                max_d = torch.FloatTensor([0.])
                min_d = torch.max(distence[idx])
                # 寻找相同类别视频最大距离
                for j in class_sim_idx[i]:
                    if j!=idx:
                        # 不是视频自己
                        max_d = torch.max(distence[idx][j],max_d)
                # 寻找不同类别当中视频最小距离
                for j in range(n):
                    if j!=idx and j not in class_sim_idx[i]:
                        min_d = torch.min(distence[idx][j],min_d)
            
            triplet_loss = triplet_loss+ torch.max(max_d-min_d+0.8,torch.FloatTensor([0.]))

        return triplet_loss
    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)

        loss_abs = self.get_absloss(contrast_pairs)

        loss_triplet = self.get_tripletloss(label,contrast_pairs)[0]
        loss_total = loss_cls + 0.01 * loss_snico + 0.0005 * loss_abs + 0.005*loss_triplet

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Abs': loss_abs,
            'Loss/Triplet': loss_triplet,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict