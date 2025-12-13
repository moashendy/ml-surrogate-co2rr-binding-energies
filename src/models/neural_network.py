import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,128,64], output_dim=1):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i+1]))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
