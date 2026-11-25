import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GraphStack(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.layers = nn.ModuleList(GCNConv(a, b) for a, b in zip(widths, widths[1:]))

    def forward(self, x, adj):
        row, col = adj.nonzero(as_tuple=True)
        edges = torch.stack((col, row))
        weights = adj[row, col]
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x, edges, weights))
        x = self.layers[-1](x, edges, weights)
        return x, torch.sigmoid(x @ x.T)


class IGAE_encoder(GraphStack):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super().__init__([n_input, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3])


class IGAE_decoder(GraphStack):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super().__init__([gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input])


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
