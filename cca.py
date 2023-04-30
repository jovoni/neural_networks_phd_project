
from train import MODELS, LR_PARITY, LR_SCRATCH_CLASSIFICATION
import torch
from data import get_dataloaders
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def extract_representation(model_name, task, K, save=False):
    
    if task == "classification":
        task = "from-scratch-classification"
        lr = LR_SCRATCH_CLASSIFICATION
    else:
        task = "parity"
        lr = LR_PARITY

    model = MODELS[model_name](K)
    # Load pre-trained model freezing certain parameters
    trained_model_path = f'models/{task}_{model_name}_{K}_{lr}_128_20.pt'
    model.load_state_dict(torch.load(trained_model_path))

    all_representations = []

    with torch.no_grad():
        dataloaders = get_dataloaders("parity", K, 128)

        for X, _ in dataloaders['test']:
            representation = model.extract_representation(X)
            all_representations.append(representation)
        
        all_representations = torch.cat(all_representations, dim=0)

    if save:
        np.save(f"representations/{task}_{model_name}_{K}.npy", all_representations.numpy())

def svcca(model_names, task_names, K, explained_variance = .99):

    Xs = []
    for model_name, task in zip(model_names, task_names):
        if task == "classification":
            task = "from-scratch-classification"
        file_name = f"representations/{task}_{model_name}_{K}.npy"

        X = np.load(file_name)
        Xs.append(pca(X, explained_variance=explained_variance))

    return cca(Xs[0], Xs[1])


def cca(X1, X2):
    print(X1.shape)
    print(X2.shape)

    n_comps = min(X1.shape[1], X2.shape[1])

    print(n_comps)

    
    cca = CCA(n_components=n_comps)
    cca.fit(X1, X2)
    return cca.correlation_


def pca(X, explained_variance = .99):    
    pca = PCA(n_components=512)
    pca.fit(X)

    i = 0
    explained_v = 0
    while (explained_v < explained_variance):
        explained_v += pca.explained_variance_ratio_[i]
        i += 1

    X_transformed = pca.transform(X)[:,:i]
    return X_transformed


if __name__ == "__main__":
    # Save representations
    # for m in MODELS.keys():
    #     for task in ["parity", "classification"]:
    #         for k in [1,3]:
    #             extract_representation(m, task, k, save=True)
    
    # Same model different task
    for m in MODELS.keys():
        for k in [1,3]:
            result = svcca([m,m], ['parity', 'classification'], k, explained_variance=.9)
            print( m, k, result)

    # Same task different models

