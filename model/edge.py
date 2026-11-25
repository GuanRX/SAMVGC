import torch
import torch.nn as nn
import torch.nn.functional as F


def degree_matirx(a):
    degree = torch.sum(a, dim=1)
    degree_mat = torch.diag(degree)
    return degree_mat


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class EdgeRedefine(nn.Module):
    def __init__(self, n_samples, topk):
        super(EdgeRedefine, self).__init__()
        self.n_samples = n_samples
        self.topk = topk
        self.w = nn.Parameter(torch.randn(n_samples * 2, 2))
        self.layer_norm = nn.LayerNorm(2)
        self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.attention = Attention(n_samples)


    def forward(self, z: torch.Tensor, a: torch.Tensor):
        # z = F.normalize(z, dim=1, p='fro')
        sim = z @ z.T / (torch.norm(z, p='fro') * torch.norm(z.T, p='fro'))     # 相似度矩阵

        # 下面几步实际上是找除自己外和自己最相似的样本
        sim = sim * (1 - torch.eye(sim.shape[0]).to(z.device))
        # max_idx = torch.argmax(sim, dim=1, keepdim=True)
        topk_sim, topk_idx = torch.topk(sim, k=1, dim=1)
        g = torch.zeros_like(sim).to(z.device)
        g.scatter_(1, topk_idx, topk_sim)

        # 求对称的g
        g = g * (1 - torch.eye(g.shape[0]).to(z.device)) + torch.eye(g.shape[0]).to(z.device)
        g = torch.maximum(g, g.T)

        d = degree_matirx(a)       # degree matrix
        a_z = torch.inverse(d) @ g

        # 求权重v
        a_f, att = self.attention(torch.stack((a_z, a), dim=1))
        # a_c = torch.cat((a_z, a), dim=1)
        # V = self.lrelu(a_c @ self.w)
        # V = self.softmax(V)
        # V = self.layer_norm(V)
        # v_z, v = torch.split(V, [1, 1], dim=1)

        # # 加权
        # ones = torch.ones((1, self.n_samples)).to(z.device)
        # a_f = v_z @ ones * a_z + v @ ones * a
        return a_f


if __name__ == '__main__':
    n_samples = 1000
    n_dims = 100
    device = torch.device('cuda:0')
    er = EdgeRedefine(n_samples).to(device)
    z = torch.randn(n_samples, n_dims).to(device)
    a = torch.rand(n_samples, n_samples).to(device)
    a_f = er(z, a)

