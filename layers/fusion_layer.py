'''
@Project: DFP-GNN
@File   : fusion_layer
@Time   : 2021/8/26 21:16
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Definition of fusion layer
'''
import torch
import torch.nn as nn
from model.attention import Attention


class FusionLayer(nn.Module):

    def __init__(self, in_channel, num_views=5, fusion_type='weighted', n_samples=None):
        """
        :param fusion_type: include concatenate/average
        """
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views

        # define the attention weights for feature matrix and adjacent matrix
        self.pai_fea = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)
        self.pai_adj = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)
        self.attention = Attention(in_channel)
        if fusion_type == "elementwise":
            self.feature_weights = nn.Parameter(torch.full(
                (num_views, n_samples, in_channel), 1.0 / num_views
            ))

    def forward(self, features):
        if self.fusion_type == "elementwise":
            return (self.feature_weights * torch.stack(features)).sum(dim=0)
        if self.fusion_type == "concat":
            combined_feature = torch.cat(features, dim=1)
        elif self.fusion_type == "average":
            combined_feature = torch.sum(features, dim=1)
        elif self.fusion_type == "weighted":
            # combine the feature matrix
            exp_sum_pai_fea = 0
            for i in range(self.num_views):
                exp_sum_pai_fea += torch.exp(self.pai_fea[i])
            combined_feature = (torch.exp(self.pai_fea[0]) / exp_sum_pai_fea) * features[0]
            for i in range(1, self.num_views):
                combined_feature = combined_feature + (torch.exp(self.pai_fea[i]) / exp_sum_pai_fea) * features[i]
        elif self.fusion_type == "attention":
            features = torch.stack(features, dim=1)
            combined_feature, att = self.attention(features)
        else:
            print("Please using a correct fusion type")
            exit()

        return combined_feature #, combined_adjacent