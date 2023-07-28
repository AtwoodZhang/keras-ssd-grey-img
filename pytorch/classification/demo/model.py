import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(OrderedDict([
      ('conv')  
    ]))