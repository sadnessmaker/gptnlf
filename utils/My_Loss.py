import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import fft, ifft, rfft, irfft
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))


