import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Conv2D -> BN (optional) -> ReLU
# Note that ReLU6 is not supported when fusing modules for QAT.
# Also note that fusing modules for QAT requires bias = False.
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
    if use_bn:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=False))
        ]))
    else:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            ('relu', nn.ReLU(inplace=False))
        ]))

# DepthwiseConv2D -> BN (optional) -> ReLU
# Note that ReLU6 is not supported when fusing modules for QAT.
# Also note that fusing modules for QAT requires bias = False.
def dw_bn_relu(in_channels, channel_multiplier=1, kernel_size=3, stride=1, padding=1, use_bn=True):
    out_channels = in_channels * channel_multiplier
    if use_bn:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=False))
        ]))
    else:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)),
            ('relu', nn.ReLU(inplace=False))
        ]))
  
def _xavier_init_(m: nn.Module):
  if isinstance(m, nn.Conv2d):
    nn.init.xavier_uniform_(m.weight) 

# Test Classification Model
class ROMClassification(nn.Module):
    def __init__(self, num_classes=10, input_size=(120,160), quantize=True, mode='train'):
        super(ROMClassification, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.quantize = quantize
        self.mode = mode
        # initialize module layers
        self.layer1 = dw_bn_relu(1, 8, kernel_size=3, stride=1, padding=1, use_bn=True)
        self.layer2 = dw_bn_relu(8, 1, kernel_size=3, stride=2, padding=1, use_bn=True)
        self.layer3 = conv_bn_relu(8, 8, kernel_size=1, stride=1, padding=0, use_bn=True)
        self.layer4 = dw_bn_relu(8, 1, kernel_size=3, stride=2, padding=1, use_bn=False)
        self.layer5 = conv_bn_relu(8, 8, kernel_size=1, stride=1, padding=0, use_bn=False)
        self.layer6 = conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=1, use_bn=True)
        self.layer7 = conv_bn_relu(8, 24, kernel_size=3, stride=2, padding=1, use_bn=False)
        self.layer8 = dw_bn_relu(24, 2, kernel_size=3, stride=1, padding=1, use_bn=False)
        self.layer9 = nn.Conv2d(48, 40, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer10 = nn.MaxPool2d(2, stride=2, padding=(1,0))
        self.layer11 = dw_bn_relu(40, 1, kernel_size=3, stride=1, padding=1, use_bn=False)
        self.layer12 = nn.Conv2d(40, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer13 = nn.MaxPool2d(2, stride=2)
        self.layer14 = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer15 = nn.Conv2d(40, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer16 = nn.Linear(320, 64)
        self.layer17 = nn.Linear(64, 64)
        self.layer18 = nn.Linear(64, 64)
        self.layer19 = nn.Linear(64, 32)
        self.layer20 = nn.Linear(320, 32)
        self.layer21 = nn.Linear(64, 10)
        # quantization needs some additional things
        if self.quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            # need to FloatFunctional for quantization compatibility
            self.concat_1 = nn.quantized.FloatFunctional()
            self.concat_2 = nn.quantized.FloatFunctional()
            self.add_1 = nn.quantized.FloatFunctional()
            self.mul_1 = nn.quantized.FloatFunctional()
            self.layers_to_fuse = [   # list of layers to fuse together (e.g. conv -> batchnorm -> relu)
                [ 'layer1.conv', 'layer1.bn', 'layer1.relu' ],
                [ 'layer2.conv', 'layer2.bn', 'layer2.relu' ],
                [ 'layer3.conv', 'layer3.bn', 'layer3.relu' ],
                [ 'layer6.conv', 'layer6.bn', 'layer6.relu' ]
            ]

    def init(self):
        self.apply(_xavier_init_)

    def forward(self, x):
        # QuantStub
        if self.quantize:
            x = self.quant(x)
        # First sequential part
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        pool_out = self.layer10(x)
        # Branching part 1
        x1 = self.layer11(pool_out)
        x1 = self.layer12(x1)
        x1 = self.layer13(x1)
        x1 = self.layer14(x1)
        x1 = torch.reshape(x1, (x1.size(0), -1))
        x2 = self.layer15(pool_out)
        x2 = torch.flatten(x2, start_dim=1)
        if self.quantize:
            x = self.concat_1.cat([x1, x2], dim=-1)
        else:
            x = torch.cat([x1, x2], dim=-1)
        # Branching part 2
        x1 = F.relu(self.layer16(x))
        x1a = F.relu(self.layer17(x1))
        if self.quantize:
            x1 = self.add_1.add(x1, x1a)
        else:
            x1 = torch.add(x1, x1a)
        x1b = self.layer18(x1)
        if self.quantize:
            x1 = self.mul_1.mul(x1, x1b)
        else:
            x1 = torch.mul(x1, x1b)
        x1 = F.relu(self.layer19(x1))
        x2 = self.layer20(x)
        if self.quantize:
            x = self.concat_2.cat([x1, x2], dim=-1)
        else:
            x = torch.cat([x1, x2], dim=-1)
        # Final sequential part
        x = self.layer21(x)
        # Softmax
        x = self.dequant(x)
        out = F.softmax(x, dim=-1)
        if self.mode == 'train':
            return x, out 
        else:
            return out


