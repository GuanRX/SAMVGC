import torch
from torch import nn
from torch.nn import functional as F
from torch_clustering import PyTorchKMeans


class ContrastConsistencyLoss(nn.Module):
    def __init__(self, n_samples, n_clusters, rate):
        super().__init__()
        self.n_clusters = n_clusters
        self.rate = rate

    def forward(self, zf, zs):
        with torch.no_grad():
            kmeans = PyTorchKMeans(n_clusters=self.n_clusters, init='k-means++',
                                   distributed=True, verbose=False)
            labels = kmeans.fit_predict(zf.detach())
            distances = (zf - kmeans.cluster_centers_[labels]).square().sum(dim=1)
            groups = []
            central = torch.zeros_like(labels, dtype=torch.bool)
            for label in range(self.n_clusters):
                members = torch.where(labels == label)[0]
                if members.numel():
                    count = max(1, int(self.rate * members.numel()))
                    selected = members[distances[members].argsort()[:count]]
                    central[selected] = True
                    groups.append(selected)

        def prototypes(features):
            return F.normalize(torch.stack([features[group].mean(0) for group in groups]))

        def log_neighbors(features):
            unit = F.normalize(features)
            return F.log_softmax(unit[~central] @ unit[central].T, dim=1)

        fused_prototypes = prototypes(zf)
        fused_log = log_neighbors(zf)
        contrast = zf.new_zeros(())
        consistency = zf.new_zeros(())
        for view in zs:
            scores = prototypes(view) @ fused_prototypes.T
            positive = scores.diagonal()
            denominator = torch.cat((positive[:, None], scores), dim=1).logsumexp(1)
            contrast = contrast + (denominator - positive).mean()
            if fused_log.size(0):
                view_log = log_neighbors(view)
                consistency = consistency + 0.5 * (
                    (view_log.exp() - fused_log.exp()) * (view_log - fused_log)
                ).sum(1).mean()
        return contrast, consistency
