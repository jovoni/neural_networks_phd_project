
import torch

def use_gpu_if_possible():
    if torch.backends.mps.is_available(): return 'mps'
    if torch.cuda.is_available(): return 'cuda'
    return 'cpu'
