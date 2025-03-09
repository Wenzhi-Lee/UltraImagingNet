import torch
import torch.nn as nn
import numpy as np

class MyLoss(nn.Module):
    def __init__(self, lambda_coord=2, lambda_r=0.2, lambda_noobj=0.1):
        super(MyLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_r = lambda_r
        self.lambda_noobj = lambda_noobj

        self.xy_loss_hist = []
        self.r_loss_hist = []
        self.coord_loss_hist = []
        self.condobj_loss_hist = []
        self.condnoobj_loss_hist = []
        self.loss_hist = []
    
    def forward(self, pred, label):
        # pred: (n, 4, 8, 8), (x, y, r, cond)
        # label: (n, defeats, 5), (row, col, x, y, r)

        batch_size = pred.shape[0]

        xy_loss = 0
        r_loss = 0
        coord_loss = 0
        condobj_loss = 0
        condnoobj_loss = 0
        for b in range(batch_size):
            predb = pred[b].permute(1, 2, 0)
            # obj loss
            for y in label[b]:
                label_row, label_col, label_x, label_y, label_r = y

                pred_x, pred_y, pred_r, pred_cond = predb[int(label_row)][int(label_col)]
                
                # coord loss
                dist = (label_x - pred_x) ** 2 + (label_y - pred_y) ** 2

                xy_loss += dist * self.lambda_coord
                r_loss += (label_r - pred_r) ** 2 * self.lambda_r

                # cond loss
                condobj_loss += (pred_cond - 1) ** 2

            # noobj loss
            obj_mask = torch.zeros(8, 8)
            for y in label[b]:
                boxRow, boxCol = y[0], y[1]
                obj_mask[int(boxRow)][int(boxCol)] = 1
            for i in range(8):
                for j in range(8):
                    if obj_mask[i][j] == 0:
                        condnoobj_loss += predb[i][j][3] ** 2 * self.lambda_noobj

        xy_loss /= batch_size
        r_loss /= batch_size
        condobj_loss /= batch_size
        condnoobj_loss /= batch_size

        coord_loss = xy_loss + r_loss
        cond_loss = condobj_loss + condnoobj_loss

        self.xy_loss_hist.append(xy_loss.item())
        self.r_loss_hist.append(r_loss.item())
        self.coord_loss_hist.append(coord_loss.item())
        self.condobj_loss_hist.append(condobj_loss.item())
        self.condnoobj_loss_hist.append(condnoobj_loss.item())
        self.loss_hist.append(coord_loss.item() + cond_loss.item())

        return coord_loss + cond_loss
    