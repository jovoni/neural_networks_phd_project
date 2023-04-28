
import torch.nn as nn

class twoLNN(nn.Module):

    def __init__(self, K):
        # K inidicating the number of MNIST images

        super(twoLNN, self).__init__()

        self.K = K
        self.layer1 = nn.Linear(28 * 28 * K, 512)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 28*28*self.K)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return(x)
