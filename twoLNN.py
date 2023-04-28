
import torch.nn as nn

class twoLNN(nn.Module):

    def __init__(self, K):
        # K inidicating the number of MNIST images

        super(twoLNN, self).__init__()
        self.K = K

        self.first_section = nn.Sequential(
            nn.Linear(28 * 28 * K, 512),
            nn.ReLU()
        )

        self.parity = nn.Linear(512, 1)
        self.classification = nn.Linear(512,10)

    def forward(self, x):
        x = x.view(-1, 28*28*self.K)
        x = self.first_section(x)
        x = self.parity(x)
        return(x)
    
    def classify(self, x):
        x = x.view(-1, 28*28*self.K)
        x = self.first_section(x)
        x = self.classification(x)
        return(x)
    

class fiveLNN(nn.Module):

    def __init__(self, K):
        # K inidicating the number of MNIST images

        super(fiveLNN, self).__init__()
        self.K = K

        self.first_section = nn.Sequential(
            nn.Linear(28 * 28 * K, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.parity = nn.Linear(64, 1)
        self.classification = nn.Linear(64,10)

    def forward(self, x):
        x = x.view(-1, 28*28*self.K)
        x = self.first_section(x)
        x = self.parity(x)
        return(x)
    
    def classify(self, x):
        x = x.view(-1, 28*28*self.K)
        x = self.first_section(x)
        x = self.classification(x)
        return(x)