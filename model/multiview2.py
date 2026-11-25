import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from .gcn import IGAE_encoder, IGAE_decoder
from .edge import EdgeRedefine
# from layers.mlfpn_gcn import MLFPN_GCN
from layers.fusion_layer import FusionLayer
from layers.cluster_layer import ClusterLayer, ClusteringHead

class SelfExpressiveLayer(nn.Module):
    def __init__(self, num_samples, init_affinity=None, device='cpu'):
        super(SelfExpressiveLayer, self).__init__()
        self.init_affinity = init_affinity
        self.device = device
        self.affinity_mat = nn.Parameter(torch.randn(num_samples, num_samples), requires_grad=True).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_affinity is not None:
            if not isinstance(self.init_affinity, torch.Tensor):
                self.init_affinity = torch.from_numpy(self.init_affinity).float()
            self.affinity_mat.data = self.init_affinity
        # else:
        #     nn.init.constant_(self.affinity_mat, 1.)

    def forward(self, x):
        latent_recon = torch.matmul(self.affinity_mat, x)
        return latent_recon


class MLP_AG(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_AG, self).__init__()
        self.w = nn.Linear(n_mlp,2)
    def forward(self, mlp_in):
        weight_output1 = F.softmax(F.leaky_relu(self.w(mlp_in)), dim=1) 
        return weight_output1
    
class SingleViewModel(nn.Module):
    def __init__(self, in_channel, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_samples, n_clusters, use_se, topk, device):
        super(SingleViewModel, self).__init__()
        self.use_se = use_se
        self.encoder = IGAE_encoder(gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, in_channel)
        self.decoder = IGAE_decoder(gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, in_channel)
        self.self_expr_layer = SelfExpressiveLayer(n_samples, init_affinity=None, device=device)

        self.cluster_layer = ClusteringHead(n_clusters, gae_n_enc_3)
        self.mlp = nn.Sequential(
            nn.Linear(gae_n_enc_3, gae_n_enc_3),
            nn.BatchNorm1d(gae_n_enc_3),
            nn.Sigmoid(),
            nn.Dropout(0.3),
        )
        self.topk = topk
        self.register_buffer("has_graph", torch.tensor(False))
        self.register_buffer("previous_graph", torch.zeros(n_samples, n_samples))
        self.register_buffer("neighbor_graph", torch.zeros(n_samples, n_samples))
        self.agcn_ag = MLP_AG(2 * n_samples)

    def current_graph(self, initial):
        if not self.has_graph:
            return initial
        pair = torch.cat((self.neighbor_graph, self.previous_graph), dim=1)
        weights = F.normalize(self.agcn_ag(pair), dim=1)
        return weights[:, :1] * self.neighbor_graph + weights[:, 1:] * self.previous_graph

    @torch.no_grad()
    def update_graph(self, initial, embedding):
        previous = self.current_graph(initial)
        unit = F.normalize(embedding, dim=1)
        similarity = unit @ unit.T
        similarity.fill_diagonal_(-torch.inf)
        values, indices = similarity.topk(min(self.topk, embedding.size(0) - 1), dim=1)
        graph = torch.zeros_like(similarity).scatter_(1, indices, values)
        graph = torch.maximum(graph, graph.T)
        graph.fill_diagonal_(1)
        self.previous_graph.copy_(previous)
        self.neighbor_graph.copy_(graph / graph.sum(1, keepdim=True).clamp_min(1e-12))
        self.has_graph.fill_(True)

    def forward(self, x, adj):
        adj = self.current_graph(adj)
        feat, z_igae_adj = self.encoder(x, adj)
        if self.use_se:
            latent_recon = self.self_expr_layer(feat)
            # latent_drop = self.mlp(latent_recon)
            recon, z_hat_adj  = self.decoder(latent_recon, adj)
        else:
            latent_recon = None
            # latent_drop = self.mlp(feat)
            recon, z_hat_adj = self.decoder(feat, adj)
        q = self.cluster_layer(feat)
        return feat, recon, latent_recon, z_hat_adj, q
    

class MultiViewModel(nn.Module):

    def __init__(self, in_channels, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, 
                 n_samples, n_clusters, fusion_type, use_se, topk, device):
        super(MultiViewModel, self).__init__()
        self.aes = []
        for in_c in in_channels:
            self.aes.append(SingleViewModel(in_c, 
                                            gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, 
                                            gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, 
                                            n_samples, n_clusters, use_se, topk, device))
        self.aes = nn.ModuleList(self.aes)

        # 新加的模块
        self.hidden_dims = gae_n_enc_3
        # self.full_graph_module = IGAE(gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, self.hidden_dims)
        # define the fusion module
        self.fusion_module = FusionLayer(self.hidden_dims, len(in_channels), fusion_type, n_samples)
        # define the clustering layer
        self.cluster_layer = ClusteringHead(n_clusters, self.hidden_dims)


    def forward(self, x, adj):
        output = []
        hidden_pr_list = []
        qs = []
        for i in range(len(x)):
            feat, recon, latent_recon, z_hat_adj, prob = self.aes[i](x[i], adj[i])
            hidden_pr_list.append(feat)
            qs.append(prob)
            output.append([feat, recon, latent_recon, z_hat_adj])

        combined_feature_pr = self.fusion_module(hidden_pr_list) # 原始view-specific graph的混合graph

        q = self.cluster_layer(combined_feature_pr)
        return output, qs, [combined_feature_pr, q]
    
