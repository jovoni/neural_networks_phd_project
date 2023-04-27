
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
        X = self.data[idx].float().flatten()
        y = self.targets[idx].float()
        return X, y
    
def get_dataloaders(K, batch_size):
    data_sets = {
        "train": datasets.MNIST(root="~/datasets/mnist", train=True, download=True),
        "test": datasets.MNIST(root="~/datasets/mnist", train=False, download=True)
        }
    
    dataloaders = {
        "train" : DataLoader(dataset=MnistKDataset(data = data_sets['train'], K=K), batch_size=batch_size), 
        "test"  : DataLoader(dataset=MnistKDataset(data=data_sets['test'], K=K), batch_size=batch_size)
    }

    return dataloaders

def load_data(ntrain, ntest):
    modes = ["train", "test"]

    data_sets = {
        "train": datasets.MNIST(root="~/datasets/mnist", train=True, download=True),
        "test": datasets.MNIST(root="~/datasets/mnist", train=False, download=True)
        }
    
    if ntest is None:
        ntest = len(data_sets["test"])

    if ntrain is None:
        ntrain = len(data_sets["train"])

    num_samples = {"train": ntrain, "test": ntest}
    
    # Prepare data
    xs = dict()
    ys = dict()

    for mode in modes:
        xs[mode] = data_sets[mode].data[:num_samples[mode]].float()
        ys[mode] = data_sets[mode].targets[:num_samples[mode]].float()
        
        mean, std = (torch.mean(xs[mode]), torch.std(xs[mode]))
        xs[mode] = (xs[mode] - mean) / std

    return xs, ys

def sample_input(xs, ys, K, mode):
    tensors = list()
    label_sum = 0

    n = len(xs[mode])
    
    for _ in range(K):
        idx = random.randint(0, n-1)

        tensors.append(xs[mode][idx])
        label_sum += ys[mode][idx]

    out = torch.cat(tensors, 1)
    #label = 2 * torch.fmod(label_sum, 2) - 1
    label = label_sum % 2

    return out, label

def load_batch(xs, ys, batch_size, K, mode):

    X = list()
    Y = list()

    for _ in range(batch_size):
        x, y = sample_input(xs, ys, K, mode)
        X.append(x.flatten())
        Y.append(y)

    X = torch.stack(X)
    Y = torch.stack(Y)

    return X, Y