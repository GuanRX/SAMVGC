import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch_geometric.nn import GCNConv
# print(torch.__version__)


class GNNLayer(Module):

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        # self.fc = nn.Linear(out_features, out_features)
        # self.bias = nn.Parameter(torch.FloatTensor(out_features))

    def forward(self, features, adj, active=False):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = self.act(output)
        return output
    
    def __str__(self) -> str:
        return f"GNNLayer(in_features={self.in_features}, out_features={self.out_features})"
    
    def __repr__(self):
        return self.__str__()


# class IGAE_encoder(nn.Module):
#     '''3层版'''

#     def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
#         super(IGAE_encoder, self).__init__()
#         self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
#         self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
#         self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
#         self.s = nn.Sigmoid()

#     def forward(self, x, adj):
#         z = self.gnn_1(x, adj, active=True)
#         z = self.gnn_2(z, adj, active=True)
#         z_igae = self.gnn_3(z, adj, active=True)
#         z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
#         return z_igae, z_igae_adj
    

# class IGAE_decoder(nn.Module):
#     '''3层版'''

#     def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
#         super(IGAE_decoder, self).__init__()
#         self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
#         self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
#         self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
#         self.s = nn.Sigmoid()

#     def forward(self, z_igae, adj):
#         z = self.gnn_4(z_igae, adj, active=True)
#         z = self.gnn_5(z, adj, active=True)
#         z_hat = self.gnn_6(z, adj, active=True)
#         z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
#         return z_hat, z_hat_adj

# class IGAE_encoder(nn.Module):
#     '''2层版'''

#     def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
#         super(IGAE_encoder, self).__init__()
#         self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
#         self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_3)
#         self.s = nn.Sigmoid()

#     def forward(self, x, adj):
#         z = self.gnn_1(x, adj, active=True)
#         z_igae = self.gnn_2(z, adj, active=False)
#         z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
#         return z_igae, z_igae_adj
    

# class IGAE_decoder(nn.Module):
#     '''2层版'''

#     def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
#         super(IGAE_decoder, self).__init__()
#         self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_3)
#         self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
#         self.s = nn.Sigmoid()

#     def forward(self, z_igae, adj):
#         z = self.gnn_4(z_igae, adj, active=True)
#         z_hat = self.gnn_6(z, adj, active=True)
#         z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
#         return z_hat, z_hat_adj


class IGAE_encoder(nn.Module):
    '''1层版'''

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_igae = self.gnn_1(x, adj, active=True)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj
    

class IGAE_decoder(nn.Module):
    '''1层版'''

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_6 = GNNLayer(gae_n_dec_1, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_hat = self.gnn_6(z_igae, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class IGAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

    def forward(self, x, adj):
        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat

    def __str__(self):
        return f"IGAE(encoder={self.encoder}, decoder={self.decoder})"

    def __repr__(self):
        return self.__str__()
