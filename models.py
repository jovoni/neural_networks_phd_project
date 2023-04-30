
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
    
    def extract_representation(self, x):
        x = x.view(-1, 28*28*self.K)
        x = self.first_section(x)
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
    
    def extract_representation(self, x):
        x = x.view(-1, 28*28*self.K)
        x = self.first_section(x)
        return(x)
    
class LeNet(nn.Module):
    def __init__(self, K):
        super(LeNet, self).__init__()

        self.K = K
        
        self.first_section = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*(5+7*K-7), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.parity = nn.Linear(84,1)
        self.classification = nn.Linear(84,10)

    def forward(self,x):        
        x = self.first_section(x)
        x = self.parity(x)
        return(x)
    
    def classify(self, x):
        x = self.first_section(x)
        x = self.classification(x)
        return(x)
    
    def extract_representation(self, x):
        x = self.first_section(x)
        return(x)