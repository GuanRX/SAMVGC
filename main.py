import os
import tqdm
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from time import time
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

from dataset import MMDataset, get_config
from utils.evaluation import cluster_accuracy, get_parameter_number, purity_score
from utils.postprocess import spixel_to_pixel_labels, affinity_to_pixellabels
from torch_clustering import PyTorchKMeans
from model.multiview2 import MultiViewModel
from model.consistency_loss import Consist_net
from model.contrast_consist import ContrastConsistencyLoss


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = get_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'))
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
if config['dataset_name'] != 'Trento':
    color_tabel = np.array([[0, 0, 0],
                            [110, 240, 70], [0, 0, 243], [230, 50, 30], [70, 157, 71], [230, 50, 240],
                            [255, 255, 84], [83, 183, 223], [227, 203, 244], [130, 80, 180],
                            [10, 35, 78], [170, 50, 100], [113, 180, 50], [200, 100, 60],
                            [60, 128, 161], [210, 146, 60], [84, 185, 54], [122, 75, 50], ])
else:
    color_tabel = np.array([[0, 0, 0],
                            [56, 108, 53], [228, 55, 38], [230, 240, 85], [170, 228, 49],
                            [147, 75, 48], [0, 204, 0], ])

color_tabel = color_tabel / 255.0

def visualization(points, shape, true, pred, save_path=None):
    fig_true = np.zeros((shape[0], shape[1], 3))
    fig_pred = np.zeros((shape[0], shape[1], 3))
    colors_true = [color_tabel[true[i]] for i in range(true.shape[0])]
    colors_pred = [color_tabel[pred[i]] for i in range(pred.shape[0])]
    fig_true[points] = colors_true
    fig_pred[points] = colors_pred


    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(fig_true)
    ax1.set_title('true')
    ax1.axis('off')
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(fig_pred) 
    ax2.set_title('predict')
    ax2.axis('off')
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print('### Save figure in {}. ###'.format(save_path))
    plt.close()


def evaluate(y_pre_sp, association_mat, gt):
    if isinstance(y_pre_sp, torch.Tensor):
        y_pre_sp = y_pre_sp.cpu().numpy()
    if isinstance(association_mat, torch.Tensor):
        association_mat = association_mat.cpu().numpy()
    y_pre_pixel = spixel_to_pixel_labels(y_pre_sp, association_mat)
    class_map = y_pre_pixel.reshape(gt.shape)

    indx = np.where(gt != 0)
    y_target = gt[indx]
    y_predict = class_map[indx]
    y_predict, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_target, y_predict, return_aligned=True)
    return (acc, kappa, nmi, ari, pur), ca, (indx, y_target.astype(int), y_predict)


def print_table(data, headers, col_width=12):
    num_cols = len(headers)
    col_spacing = 3
    
    # create row format
    row_format = "".join(["{:>"+str(col_width)+"}"]*(num_cols))
    row_separator = "".join(["{:->"+str(col_width)+"}"]*(num_cols))
    
    # print header
    print(row_format.format(*headers))
    print(row_separator.format(*["" for _ in range(num_cols)]))
    
    # print data rows
    for i, row in enumerate(data):
        row_values = [f"{val:.4f}" if isinstance(val, float) else val for val in row]
        print(row_format.format(*row_values))


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train(config, pretrain=True):
    dataset_name = config['dataset_name']
    dataset = MMDataset(
        dataset_name, 
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
    if config['remove_bkg']:
        dataset.remove_background()
    n_clusters = dataset.n_classes
    n_samples = len(dataset)
    lr = config['lr']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = config['n_epoches']
    
    gae_n_enc_1, gae_n_enc_2, gae_n_enc_3 = config['gae_n_enc_1'], config['gae_n_enc_2'], config['gae_n_enc_3']
    gae_n_dec_1, gae_n_dec_2, gae_n_dec_3 = config['gae_n_dec_1'], config['gae_n_dec_2'], config['gae_n_dec_3']
    refine_k, rate = config[dataset_name]['refine_k'], config[dataset_name]['rate']     # 控制簇内对比学习比例
    alpha_recon, alpha_consist, alpha_contrast = config[dataset_name]['alpha_recon'], config[dataset_name]['alpha_consist'], config[dataset_name]['alpha_contrast']

    fusion_type = config['fusion_type']
    use_se = True       # 使用自表达层

    save_path = config['save_path'] + '/' + dataset_name + '.pkl'
    n_views = len(dataset.x)

    x = [torch.from_numpy(it.reshape(n_samples, -1)).float().to(device) for it in dataset.x]
    # x.append(torch.cat(x, dim=1))
    adj = [torch.from_numpy(it).float().to(device) for it in dataset.adj]

    y = dataset.y
    model = MultiViewModel(
        in_channels=[it.shape[1] for it in x],
        gae_n_enc_1=gae_n_enc_1,
        gae_n_enc_2=gae_n_enc_2,
        gae_n_enc_3=gae_n_enc_3,
        gae_n_dec_1=gae_n_dec_1,
        gae_n_dec_2=gae_n_dec_2,
        gae_n_dec_3=gae_n_dec_3,
        n_samples=n_samples,
        n_clusters=n_clusters,
        fusion_type=fusion_type,
        use_se=use_se,
        topk=refine_k,
        device=device).to(device)
    
    get_parameter_number(model)


    optimizer = Adam(model.parameters(), lr=lr)
    kmeans = PyTorchKMeans(n_clusters=n_clusters, init='k-means++', distributed=True, verbose=False)


    contrast_consist = ContrastConsistencyLoss(n_samples, n_clusters, rate).to(device)

    # 直接用原数据测试一下
    z = torch.cat(x, dim=1)         # 不同模态cat起来的
    y_pred = kmeans.fit_predict(z).cpu().numpy()
    if config['remove_bkg']:
        y_pred = dataset.recover_background(y_pred)
    mts, ca, eta = evaluate(y_pred, dataset.association_mat, dataset.gt)
    print('[initial accuracy]')
    print_table([mts], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
    print()

    t0 = time()
    if pretrain:
        print('start training.')
        best_acc = [0] * 5
        for epoch in range(1, EPOCH + 1):
            model.train()
            output, qs, combined = model(x, adj)    # output: feat, recon, latent_recon, adj_hat
            combined_feature, q = combined

            recon_loss = 0

            for v in range(n_views):
                recon_loss += F.mse_loss(output[v][1], x[v])
                # recon_loss += F.mse_loss(output[v][0], output[v][2])
                if use_se:
                    recon_loss += F.mse_loss(output[v][0], output[v][2])
            contrast_loss, clu_consist_loss = contrast_consist(combined_feature, [output[v][0] for v in range(n_views)])

            loss = 0
            loss += alpha_recon * recon_loss

            loss += alpha_contrast * contrast_loss + alpha_consist * clu_consist_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if epoch % 10 == 0:
                for view, graph, features in zip(model.aes, adj, output):
                    view.update_graph(graph, features[0].detach())
            print(f'Epoch {epoch}: loss = {loss.item()}')
            
            model.eval()
            with torch.no_grad():
                output, qs, combined = model(x, adj)
            combined_feature, q = combined
            z = torch.cat([output[v][0] for v in range(n_views)], dim=1) # 不同模态cat起来的
            y_pred = kmeans.fit_predict(z).cpu().numpy()
            if config['remove_bkg']:
                y_pred = dataset.recover_background(y_pred)
            mts1, ca, eta = evaluate(y_pred, dataset.association_mat, dataset.gt)
            if mts1[0] > best_acc[0]:
                best_acc = mts1
                weights = deepcopy(model.state_dict())
                indx, true, pred = eta
                visualization(indx, dataset.gt.shape, true, pred.astype(int), 
                              config['result_path'] + '/' + dataset_name + f'/{dataset_name}_{round(best_acc[0], 4)}.png')

            y_pred = kmeans.fit_predict(combined_feature.detach()).cpu().numpy()     # 不同模态attention融合起来的 dim=1)
            if config['remove_bkg']:
                y_pred = dataset.recover_background(y_pred)
            mts2, ca, eta = evaluate(y_pred, dataset.association_mat, dataset.gt)
            if mts2[0] > best_acc[0]:
                indx, true, pred = eta
                best_acc = mts2
                weights = deepcopy(model.state_dict())
                visualization(indx, dataset.gt.shape, true, pred.astype(int), 
                              config['result_path'] + '/' + dataset_name + f'/{dataset_name}_{round(best_acc[0], 4)}.png')

            print_table([best_acc, mts1, mts2], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
        torch.save(weights, save_path)
        print(f'best weight saved in {save_path}')
        print('finish.')

    
    return best_acc, time() - t0


def rand_seed(seed):
    if seed == -1:
        return
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # config = get_config('./config.json')
    rand_seed(config['seed'])

    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(config['log_path']):
        os.mkdir(config['log_path'])
    if not os.path.exists(config['result_path']):
        os.mkdir(config['result_path'])
    if not os.path.exists(config['result_path'] + '/' + config['dataset_name']):
        os.mkdir(config['result_path'] + '/' + config['dataset_name'])
    best_acc, t = train(config, pretrain=True)
    print()
    print_table([best_acc], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
    print(f'total time cost:{t}s', )