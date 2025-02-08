import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def defeatMatrix2Grid(defeats, gridN):
    grid = np.zeros((gridN, gridN))
    for defeat in defeats:
        x, y, r = gridN - defeat[0], defeat[1], defeat[2]
        for i in range(x - r, x + r + 1):
            for j in range(y - r, y + r + 1):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    if 0 <= i < gridN and 0 <= j < gridN:
                        grid[i, j] = 1
    return grid

def weight_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(model, nn.Parameter):
        nn.init.kaiming_normal_(model, mode='fan_out', nonlinearity='relu')