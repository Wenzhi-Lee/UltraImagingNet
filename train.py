import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
from model.defeat_detect import DefeatDetectModel
from dataset import DefectDataset, my_collate
from torch.utils.data import DataLoader
from myloss import MyLoss
import utils
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
lr = 1e-4
epochs = 30

# Data parameters
file_name = 'data/final_data_sim_{:04d}.mat'
file_num = 32
train_ratio = 0.9
scan_num = 4
gridN = 512

model = DefeatDetectModel(gridN).to(device)
model.apply(utils.weight_init)

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = MyLoss().to(device)

start_time = time.time()

train_data = DefectDataset(file_name, file_num * train_ratio)
test_data = DefectDataset(file_name, file_num - int(file_num * train_ratio), file_num * train_ratio)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, shuffle=False, collate_fn=my_collate)

for epoch in range(epochs):
    # Training
    loop = tqdm.tqdm(train_loader, leave=True)
    loop.set_description(f'Epoch {epoch + 1}/{epochs}')

    loss_history = []

    model.train()

    for phase, label in train_loader:

        phase, label = phase.to(device), label

        pred = model(phase)
        loss = loss_fn(pred, label)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix({
            'loss': '%.4f' % loss.item(),
            'time': time.time() - start_time
        })
        loop.update()

    loop.set_postfix({
        'loss': '%.4f' % np.mean(loss_history),
        'time': time.time() - start_time
    })
    loop.close()

torch.save(model.state_dict(), 'model.pth')

# Evaluation
model.eval()

prob_threshold = 0.5

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(
    max((len(test_loader) // scan_num + 3) // 4, 2), 4,
    figsize=(16, 4 * ((len(test_loader) // scan_num + 3) // 4))
)

for i, (phase, label) in enumerate(test_loader):
    if i % scan_num != 0:
        continue

    phase = phase.to(device)
    pred = model(phase)[0].detach().cpu().numpy()
    phase = phase[0].detach().cpu().numpy()

    ind = i // scan_num
    ax[ind // 4, ind % 4].imshow(phase, cmap='jet')
    ax[ind // 4, ind % 4].set_title(f' {ind}')
    ax[ind // 4, ind % 4].axis('off')

    # Draw ground truth
    boxN = 8
    boxLen = gridN / boxN
    for truth in label[0]:
        x, y, r = truth[2] * boxLen, truth[3] * boxLen, truth[4] * boxLen
        basex, basey = truth[0] * boxLen, truth[1] * boxLen
        x, y = x + basex, y + basey
        rect = patches.Circle((y, x), r, edgecolor='g', alpha=0.7)
        ax[ind // 4, ind % 4].add_patch(rect)
    
    # Draw prediction
    for i in range(boxN):
        for j in range(boxN):
            if pred[i][j][3] > prob_threshold:
                x, y, r = pred[i][j][:3] * boxLen
                basex, basey = i * boxLen, j * boxLen
                x, y = x + basex, y + basey
                rect = patches.Circle((y, x), r, edgecolor='r', lw=2, facecolor='none')
                ax[ind // 4, ind % 4].add_patch(rect)
            
            if pred[i][j][7] > prob_threshold:
                x, y, r = pred[i][j][4:7] * boxLen
                basex, basey = i * boxLen, j * boxLen
                x, y = x + basex, y + basey
                rect = patches.Circle((y, x), r, edgecolor='r', lw=2, facecolor='none')
                ax[ind // 4, ind % 4].add_patch(rect)

plt.show()