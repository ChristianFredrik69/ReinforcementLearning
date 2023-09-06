import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 25)
        self.linear2 = nn.Linear(25, 25)
        self.linear3 = nn.Linear(25, 2)
        
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x

def num_params(module):
    num_params = 0
    for params in module.parameters():
        num_params += params.numel()
    return num_params

if __name__ == '__main__':
    network = QNetwork()
    num_params = 0
    for params in network.parameters():
        print(params)
        num_params += params.numel()


    print(network(torch.zeros(4)))
    print("hallo")
    
    torch.nn.init.xavier_normal_(network.parameters())



















