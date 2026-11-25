'''
@Project: DFP-GNN
@File   : cluster_layer
@Time   : 2021/8/26 21:16
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Definition of clustering layer
'''
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ClusterLayer(nn.Module):

    def __init__(self, num_clusters, hidden_dim, alpha=1):
        """
        :param num_clusters: the number of clusters
        :param hidden_dim: the dimension of hidden layer
        :param alpha:
        """
        super(ClusterLayer, self).__init__()
        self.alpha = alpha
        self.network = Parameter(torch.Tensor(num_clusters, hidden_dim)).float()
        torch.nn.init.xavier_normal_(self.network.data)

    def forward(self, z) -> torch.Tensor:
        """
            computation in the clustering layer
        :param z: the input hidden embedding
        :return: the soft distribution of data
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.network, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q
    

class ClusteringHead(nn.Module):    # 这个类写法是正常的，不会引发inplace问题
    def __init__(self, n_class, n_dim, alpha=1.):
        super(ClusteringHead, self).__init__()
        # Clustering head
        self.alpha = alpha
        # initial_cluster_centers = torch.tensor(torch.randn((n_class, n_dim), dtype=torch.float, requires_grad=True))
        self.cluster_centers = nn.Parameter(torch.Tensor(n_class, n_dim), requires_grad=True)
        # torch.nn.init.orthogonal_(self.cluster_centers.data, gain=1)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        """
        :param x: n_batch * n-dim
        :return:
        """
        pred_prob = self.get_cluster_prob(x)
        return pred_prob

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1).clone() - self.cluster_centers.clone()) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)