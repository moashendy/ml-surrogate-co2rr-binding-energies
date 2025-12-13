from src.models.neural_network import MLPRegressor
import torch
import numpy as np

def test_mlp_forward():
    model = MLPRegressor(input_dim=4)
    x = torch.randn(2,4)
    out = model(x)
    assert out.shape == torch.Size([2])
