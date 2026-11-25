import numpy as np
import scipy.io as sio
import copy
import torch
import warnings
from sklearn.preprocessing import StandardScaler, minmax_scale, normalize, scale
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from collections import Counter

from utils.superpixel_utils import HSI_to_superpixels, create_association_mat, create_spixel_graph, show_superpixel, extract_superpixel_features

def get_config(config_path):
    import json
    fp = open(config_path, 'r')
    return json.load(fp)

def prepare_data(mat_path):
    data = sio.loadmat(mat_path)
    key = [k for k in data if k != '__version__' and k != '__header__' and k != '__globals__']
    pixels = data[key[0]]
    return pixels

def get_sp_labels(seg, gt):
    labels = np.zeros((np.unique(seg).shape[0]), dtype=np.int32)
    for i in range(len(labels)):
        mask = seg == i
        label = gt[mask]
        c = Counter(label).most_common()         # 出现频率最多的标签
        if c[0][0] == 0 and len(c) > 1:
            labels[i] = c[1][0]                 # 0是背景,计算精度会忽略
        else:
            labels[i] = c[0][0]
    return labels


class MMDataset():
    def __init__(self, dataset_name, mm_data_path, gt_path, n_pc, patch_size, rgb, n_neighbors, n_superpixels, slic_compactness=10.0) -> None:
        self.dataset_name = dataset_name
        mm_data_path = mm_data_path
        gt_path = gt_path
        n_pc = n_pc
        patch_size = patch_size

        mm_data = [prepare_data(data_path) for data_path in mm_data_path]
        for i in range(len(mm_data)):
            if len(mm_data[i].shape) == 2:
                mm_data[i] = mm_data[i][:, :, np.newaxis]   # 防止某些模态只有2D
        gt = prepare_data(gt_path)

        n_row, n_column, n_band = mm_data[0].shape  # mm_data[0]为HSI
        hsi = copy.deepcopy(mm_data[0])             # 保存一个HSI的副本
        
        if n_pc is not None:
            pca = PCA(n_components=n_pc)
            mm_data[0] = minmax_scale(pca.fit_transform(mm_data[0].reshape(n_row * n_column, n_band))).reshape((n_row, n_column, -1))
        else:
            mm_data[0] = minmax_scale(mm_data[0].reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))

        if rgb is None:
            warnings.warn('No matched dataset name. RGB bands set to [0, 1, 2].')
            rgb = [0, 1, 2]
        anchor_features = np.concatenate([
            minmax_scale(view.reshape(n_row * n_column, -1)).reshape(view.shape)
            for view in mm_data
        ], axis=-1)
        sp_labels = HSI_to_superpixels(anchor_features, n_superpixels=n_superpixels, compactness=slic_compactness)
        show_superpixel(sp_labels, hsi[:, :, rgb], 'result/' + self.dataset_name + '_sp.pdf')       # 需要特定通道在第三维上修改，如img[:, :, (10, 20, 30)]
        self.association_mat = create_association_mat(sp_labels)
        self.association_mat_copy = self.association_mat[:]      # 保存一个副本
        x = []
        adj = []
        for i, data in enumerate(mm_data):
            sp_features = extract_superpixel_features(data, sp_labels, mode='center_patch', patch_size=patch_size)   # 取patch的超像素特征
            # _, adj_wave, adj_hat, norm, weight_tensor = load_single_view_data(sp_features, self.dataset_name, None, i, False, n_neighbors, True, True, 2)
            # fadj = kneighbors_graph(sp_features.reshape(sp_features.shape[0], -1), n_neighbors=n_neighbors, mode='distance', include_self=False).toarray()
            fadj, sadj, sp_centroid = create_spixel_graph(data, sp_labels, n_neighbors)
            x.append(sp_features)
            # edge_index, edge_weight = from_scipy_sparse_matrix(fadj)
            adj.append(fadj)

        self.x = x
        self.adj = adj
        self.y = get_sp_labels(sp_labels, gt)             
        self.n_classes = np.unique(self.y).shape[0] - 1        # 去掉背景0
        self.gt = gt

    def remove_background(self):        # 去掉超像素标签为0的超像素，以防干扰
        non_zero_idx = np.nonzero(self.y)[0]
        self.association_mat = self.association_mat[:, non_zero_idx]
        for i in range(len(self.x)):
            self.x[i] = self.x[i][non_zero_idx, :]
            self.adj[i] = self.adj[i][non_zero_idx, :][:, non_zero_idx]
        
    def recover_background(self, sp_pred):  # 恢复回来以进行精度计算
        if isinstance(sp_pred, torch.Tensor):
            sp_pred = sp_pred.detach().cpu().numpy()
        non_zero_idx = np.nonzero(self.y)[0]
        sp_pred_full = np.zeros_like(self.y)
        sp_pred_full[non_zero_idx] = sp_pred
        self.association_mat = self.association_mat_copy
        return sp_pred_full

    def __getitem__(self, index):                   # maybe for classification
        return self.x[:, index], self.y[:, index]

    def __len__(self):
        return self.x[0].shape[0]
    
    def __str__(self) -> str:
        string = f'========================================\n'\
                f'Dataset name: \t\t{self.dataset_name}\n'\
                f'----------------------------------------\n'\
                f'Number of classes: \t{self.n_classes}\n'\
                f'Number of real samples\t{np.where(self.y != 0)[0].shape[0]}\n'\
                f'----------------------------------------\n'\
                f'data\t\t\tshape\n'\
                f'----------------------------------------\n'
        for i in range(len(self.x)):
            string += f'x{i} \t\t\t{self.x[i].shape}\n'
            string += f'adj{i} \t\t\t{self.adj[i].shape}\n'
        string += f'y \t\t\t{self.y.shape}\n'
        string += f'========================================\n'
        return string
    
    def __repr__(self) -> str:
        return self.__str__()


if __name__ == '__main__':
    config = get_config('./config.json')
    dataset_name = config['dataset_name']
    dataset = MMDataset(dataset_name, 
                        config[dataset_name]['mm_data_path'], 
                        config[dataset_name]['gt_path'],
                        config[dataset_name]['n_pc'],
                        config[dataset_name]['patch_size'],
                        config[dataset_name]['rgb'],
                        config[dataset_name]['n_neighbors'],
                        config[dataset_name]['n_superpixels'],
                        config[dataset_name].get('slic_compactness', 10.0),
                        )
    print(dataset)
