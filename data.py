
import torch
import torchvision.datasets as datasets
import random
from torch.utils.data import DataLoader, Dataset

class MnistKDataset(Dataset):

    def __init__(self, data, K):
        indices = torch.empty(K, len(data), dtype=torch.int64)
        for k in range(K):
            indices[k] = torch.randperm(len(data))

        self.data = torch.cat([data.data[indices[i]] for i in range(K)], dim=2)
        self.targets = sum([data.targets[indices[i]] for i in range(K)]) % 2

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx].float()
        y = self.targets[idx].float()
        return X, y
    
def get_dataloaders(K, batch_size):
    data_sets = {
        "train": datasets.MNIST(root="datasets/mnist", train=True, download=True),
        "test": datasets.MNIST(root="datasets/mnist", train=False, download=True)
        }
    
    dataloaders = {
        "train" : DataLoader(dataset=MnistKDataset(data=data_sets['train'], K=K), batch_size=batch_size), 
        "test"  : DataLoader(dataset=MnistKDataset(data=data_sets['test'], K=K), batch_size=batch_size)
    }

    return dataloaders