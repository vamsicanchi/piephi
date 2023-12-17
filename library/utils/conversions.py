import torch
import numpy as np

def xyxy2xywh(x):
    
    # Transform box coordinates from [x1, y1, x2, y2] (where x1y1=top-left, x2y2=bottom-right) to [x, y, w, h] 
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    
    return y

def xywh2xyxy(x):
    
    # Get box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where x1y1=top-left, x2y2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    
    return y

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix