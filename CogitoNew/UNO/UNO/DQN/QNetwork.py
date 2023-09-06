import torch.nn as nn
import torch


class QNetwork(nn.Module):

    def __init__(self, ob_dim, action_space, hidden_dim=25):
        
        super().__init__()
        
        self.linear1 = nn.Linear(ob_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_space)
        self.activation = nn.ReLU

    def forward(self, x):
        
        x = torch.flatten(x)
        x = x.type(torch.float32)
        x = self.activation(self.linear1)
        x = self.activation(self.linear2)
        x = self.linear3(x)

        return x


def num_params(module):
    antall = 0
    for params in module.parameters():
        antall += params.numel()
    return antall


if __name__ == '__main__':
    network = QNetwork(4, 2)
    print(num_params(network))
    print(network(torch.zeros(4)))


