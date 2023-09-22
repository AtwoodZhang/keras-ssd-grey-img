import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import pdb

def layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, acti, BN, bias=False):
    if BN == True:
        return nn.Sequential(OrderedDict([
                        ( layer_name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)),
                        ('bn', nn.BatchNorm2d(out_channels)),
                        ('relu', acti),
                    ]))
    else:
        return nn.Sequential(OrderedDict([
                        ( layer_name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)),
                        ('relu', acti),
                    ]))

class LayerTest(nn.Module):
    def __init__(self, config, debug=False):
        super().__init__()

        self.debug = debug

        layer_pars = config["model"]
        self.quantize = config["model"]["quantize"]
        all_layers = ["layer_00", "layer_01", "layer_02", "layer_03", "layer_04", "layer_05", "layer_06", "layer_07", "layer_08", "layer_09", "layer_10", "layer_11", "layer_12", "layer_13"]

        self.all_layer_conv_names = []
        for id, layer_n in enumerate(all_layers):
            layer_config = layer_pars[layer_n]
            if layer_config["type"]:
                if layer_config["type"] == "DW":
                    in_channels = layer_config["in_channels"]
                    K = layer_config["K"]
                    out_channels = K * layer_config["in_channels"]
                    groups = in_channels
                    layer_name = 'dw_conv'

                else:
                    in_channels = layer_config["in_channels"]
                    out_channels = layer_config["out_channels"]
                    groups = 1
                    layer_name = 'conv'
                
                kernel_size = (layer_config["kernel"], layer_config["kernel"])
                stride = layer_config["stride"]
                padding = (layer_config["padding"]["padding_w"], layer_config["padding"]["padding_h"])
                BN = layer_config["BN"]
                activation = layer_config["activation"]
                
                if activation == "RELU6":
                    self.acti = nn.ReLU(inplace=False)
                elif activation == "RELU":
                    self.acti = nn.ReLU(inplace=False)

                if id == 0:
                    if layer_config["set"] == True:
                        self.layer_00 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 1:
                    if layer_config["set"] == True:
                        self.layer_01 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 2:
                    if layer_config["set"] == True:
                        self.layer_02 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 3:
                    if layer_config["set"] == True:
                        self.layer_03 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 4:
                    if layer_config["set"] == True:
                        self.layer_04 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 5:
                    if layer_config["set"] == True:
                        self.layer_05 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 6:
                    if layer_config["set"] == True:
                        self.layer_06 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 7:
                    if layer_config["set"] == True:
                        self.layer_07 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 8:
                    if layer_config["set"] == True:
                        self.layer_08 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 9:
                    if layer_config["set"] == True:
                        self.layer_09 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 10:
                    if layer_config["set"] == True:
                        self.layer_10 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 11:
                    if layer_config["set"] == True:
                        self.layer_11 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 12:
                    if layer_config["set"] == True:
                        self.layer_12 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                elif id == 13:
                    if layer_config["set"] == True:
                        self.layer_13 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                
                if not activation:
                    self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".bn"])
                else:
                    self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".bn", "layer_"+str(id).zfill(2)+".relu"])



        self.classification = nn.Linear(config["model"]["layer_13"]["out_channels"], config["training"]["classes"])   
        if self.quantize:
            self.quant = torch.quantization.QuantStub()
            self.quant_1 = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):

        ## Order of execution
        activations = {}
        if self.quantize:
            x = self.quant(x)
        
        activations['input_image_tensor'] = x

        # Layer_00: Depthwise Conv, BN, ReLU 6
        x = self.layer_00(x)
        activations['OP_0_DEPTHWISECONV2D'] = x

        # Layer_01: Depthwise Conv, BN, ReLU 6
        x = self.layer_01(x)
        activations['OP_1_DEPTHWISECONV2D'] = x

        # Layer_02: Depthwise Conv, BN, ReLU 6
        x = self.layer_02(x)
        activations['OP_2_DEPTHWISECONV2D'] = x

        # Layer_03: Conv 2D, BN, ReLU
        x = self.layer_03(x)
        activations['OP_3_CONV2D'] = x

        # Layer_04: Conv 2D, BN, ReLU 6
        x = self.layer_04(x)
        activations['OP_4_CONV2D'] = x
        
        # Layer_05: Conv 2D, BN,  ReLU
        x = self.layer_05(x)
        activations['OP_5_CONV2D'] = x

        # Layer_06: Conv 2D, BN
        x = self.layer_06(x)
        activations['OP_6_CONV2D'] = x

        # Reshape
        x = torch.reshape(x, (x.shape[0], 8, 24, 34))
        activations['OP_7_RESHAPE'] = x

        # Layer_08: COnv 2D, BN
        x = self.layer_08(x)
        activations['OP_8_CONV2D'] = x
        # Reshape
        x = torch.reshape(x, (x.shape[0], 24, 17, 17))
        activations['OP_9_RESHAPE'] = x

        # Layer_10: Conv 2D, BN, ReLU
        x = self.layer_10(x)
        activations['OP_10_CONV2D'] = x
        # Layer_11: Conv 2D, BN, ReLU
        x = self.layer_11(x)
        activations['OP_11_CONV2D'] = x

        # Layer_12: Conv 2D, BN, ReLU 6
        x = self.layer_12(x)
        activations['OP_12_CONV2D'] = x

        # Layer_13: Conv 2D, BN, ReLU 6
        x = self.layer_13(x)
        activations['OP_13_CONV2D'] = x

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        activations['OP_14_RESHAPE'] = x
        x = self.classification(x)
        activations['OP_15_FULLYCONNECTED'] = x
        x = self.dequant(x)

        x = F.softmax(x, dim=1)
        
        # x_1 = self.dequant(x)
        x_softmax = self.quant_1(x)
        activations['OP_16_SOFTMAX'] = x #x_1

        # x = torch.transpose(x, 0, 1)

        if self.debug:
            return x, activations
        else:
            return x



def build_model(config, debug):
    net = LayerTest(config, debug)
    return net

def prepare_model(model):
    model.train() # set up model in training mode. this does not actually perform training.
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.fuse_modules(model, model.all_layer_conv_names)   # fuse conv/bn/relu layers
    model = torch.quantization.prepare_qat(model) # convert fp32 model to quantize-aware model
    return model 