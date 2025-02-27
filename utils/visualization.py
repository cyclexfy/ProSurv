import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

from models.model_BinaryMemory import *

# Flatten the memory banks
def prepare_data(memory_bank):
    n_classes, size, dim = memory_bank.shape
    data = memory_bank.reshape(-1, dim)  # Flatten to [n_classes * size, dim]
    labels = np.repeat(np.arange(n_classes), size)  # Generate class labels
    return data, labels

# Plotting function
def plot_tsne(embedded, labels, title, save_dir, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(len(np.unique(labels))))
    # plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, filename))  # 保存图片

if __name__ == '__main__':
    binary_memory_model = BinaryMemory(omic_input_dim=4999)
    dual_memory_model = DualMemory(omic_input_dim=4999)
    memory_model = DualMemory(omic_input_dim=4999)

    fold = 0
    model = memory_model
    # model_dir = "/data/lfc/code/MultiModalLearning/MissingModality/PTCA/results/results_brca/BinaryMemory_1/tcga_brca__nll_surv_a0.5_lr1e-05_l2Weight_0.0001_5foldcv_b4_survival_months_dss_dim1_1024_patches_4096_wsiDim_256_epochs_50_fusion_concat_modality_BinaryMemory_pathT_combine/s_3_checkpoint.pt"
    model_dir = f'/data/lfc/code/MultiModalLearning/MissingModality/PTCA/results/results_brca/Memory/tcga_brca__nll_surv_a0.5_lr1e-05_l2Weight_0.0001_5foldcv_b4_survival_months_dss_dim1_1024_patches_4096_wsiDim_256_epochs_30_fusion_concat_modality_DualMemory_pathT_combine/s_{fold}_checkpoint.pt'
    # model_dir = '/data/lfc/code/MultiModalLearning/MissingModality/PTCA/results/results_brca/BinaryMemory/tcga_brca__nll_surv_a0.5_lr1e-05_l2Weight_0.0001_5foldcv_b4_survival_months_dss_dim1_1024_patches_4096_wsiDim_256_epochs_50_fusion_concat_modality_BinaryMemory_pathT_combine/s_0_checkpoint.pt'
    model.load_state_dict(torch.load(model_dir, weights_only=True))

    # memory_bank
    path_memory_bank = model.patho_memory_bank.cpu().numpy()  #  [n_classes, size, dim]
    geno_memory_bank = model.geno_memory_bank.cpu().numpy() #  [n_classes, size, dim]

    print(path_memory_bank.shape)
    print(geno_memory_bank.shape)

    # Prepare data
    path_data, path_labels = prepare_data(path_memory_bank)
    geno_data, geno_labels = prepare_data(geno_memory_bank)
    # # Apply PCA
    # pca = PCA(n_components=2)
    # path_embedded = pca.fit_transform(path_data)
    # geno_embedded = pca.fit_transform(geno_data)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, path_data.shape[0] - 1), n_iter=1000)
    path_embedded = tsne.fit_transform(path_data)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, geno_data.shape[0] - 1), n_iter=1000)
    geno_embedded = tsne.fit_transform(geno_data)

    save_dir = '/data/lfc/code/MultiModalLearning/MissingModality/PTCA/visualization/'
    # Plot as before
    plot_tsne(path_embedded, path_labels, 'PCA Visualization of Path Memory Bank', save_dir, 'path_prototype.png')
    plot_tsne(geno_embedded, geno_labels, 'PCA Visualization of Geno Memory Bank', save_dir, 'geno_prototype.png')
