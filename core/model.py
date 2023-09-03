import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

# (a) Feature Embedding and (b) Actionness Modeling
class Actionness_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Actionness_Module, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.f_embed(out)
        embeddings = out.permute(0, 2, 1)
        out = self.dropout(out)
        out = self.f_cls(out)
        cas = out.permute(0, 2, 1)
        actionness = cas.sum(dim=2)
        return embeddings, cas, actionness

# APSL Pipeline
class APSL(nn.Module):
    def __init__(self, cfg):
        super(APSL, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES

        self.actionness_module = Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_p = cfg.R_P
        self.r_n = cfg.R_N
        self.m = cfg.m
        self.M = cfg.M

        self.dropout = nn.Dropout(p=0.6)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def positive_snippets_mining(self, actionness, embeddings, k_positive):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        positive_act = self.select_topk_embeddings(actionness_drop, embeddings, k_positive)
        positive_bkg = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_positive)

        return positive_act, positive_bkg

    def negative_snippets_mining(self, actionness, embeddings, k_negative):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        negative_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_negative)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        negative_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_negative)

        return negative_act, negative_bkg

    def get_video_cls_scores(self, cas, k_positive):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_positive, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores

    def forward(self, x):
        # x:[B,750,2048]
        num_segments = x.shape[1]
        k_positive = num_segments // self.r_p
        k_negative = num_segments // self.r_n
        # embeddings:[B,750,2048] cas:[B,750,20] actionness:[16,750]
        embeddings, cas, actionness = self.actionness_module(x)
        # positive_act:[B,k_positive,2048] positive_bkg:[B,150,2048] 
        positive_act, positive_bkg = self.positive_snippets_mining(actionness, embeddings, k_positive)
        # negative_act:[B,k_negative,2048] negative_bkg:[B,k_negative,2048]
        negative_act, negative_bkg = self.negative_snippets_mining(actionness, embeddings, k_negative)
        # video_scores:[B,20]
        video_scores = self.get_video_cls_scores(cas, k_positive)

        contrast_pairs = {
            'PA': positive_act,
            'PB': positive_bkg,
            'NA': negative_act,
            'NB': negative_bkg
        }

        return video_scores, contrast_pairs, actionness, cas, embeddings
