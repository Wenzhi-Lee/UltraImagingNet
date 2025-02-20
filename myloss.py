import torch
import torch.nn as nn
import numpy as np

from lossutil import iou

class MyLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(MyLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, pred, label):
        # pred: (n, 8, 8, 8)
        # label: (n, defeats, 5)

        batch_size = pred.shape[0]

        coord_loss = 0
        cond_loss = 0
        for b in range(batch_size):
            predb = pred[b]
            for y in label[b]:
                boxRow, boxCol = y[0], y[1]
                predbox = predb[int(boxRow)][int(boxCol)]
                iou1 = iou(predbox[0], predbox[1], predbox[2], y[2], y[3], y[4])
                iou2 = iou(predbox[4], predbox[5], predbox[6], y[2], y[3], y[4])
                
                # select the circle with higher iou
                circle = predbox[:4]
                un_conf = predbox[7]
                bigger_iou = iou1
                smaller_iou = iou2
                if iou2 > iou1:
                    circle = predbox[4:]
                    un_conf = predbox[3]
                    bigger_iou = iou2
                    smaller_iou = iou1
                
                xy_loss = (circle[0] - y[2]) ** 2 + (circle[1] - y[3]) ** 2
                r_loss = (circle[2] - y[4]) ** 2

                coord_loss += (xy_loss + r_loss) * self.lambda_coord

                cond_loss += (bigger_iou - circle[3]) ** 2
                cond_loss += (smaller_iou - un_conf) ** 2 * self.lambda_noobj

            # noobj loss
            obj_mask = torch.zeros(8, 8)
            for y in label[b]:
                boxRow, boxCol = y[0], y[1]
                obj_mask[int(boxRow)][int(boxCol)] = 1
            for i in range(8):
                for j in range(8):
                    if obj_mask[i][j] == 0:
                        cond_loss += (predb[i][j][3] ** 2 + predb[i][j][7] ** 2) * self.lambda_noobj
        
        return (coord_loss + cond_loss) / batch_size