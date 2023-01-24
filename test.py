import torch 
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        nn.init.kaiming_uniform_(self.weight)

        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

m = Model()

print(m)