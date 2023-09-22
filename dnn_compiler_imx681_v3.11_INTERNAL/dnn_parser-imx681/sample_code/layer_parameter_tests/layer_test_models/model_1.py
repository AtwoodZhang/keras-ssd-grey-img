import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import pdb

def layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, acti, BN, bias=False):
    if layer_name in ["dw_conv", "conv"]:
        if BN == True and acti is not None:
            return nn.Sequential(OrderedDict([
                            ( layer_name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)),
                            ('bn', nn.BatchNorm2d(out_channels)),
                            ('relu', acti),
                        ]))
        elif BN == False and acti is not None:
            return nn.Sequential(OrderedDict([
                            ( layer_name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)),
                            ('relu', acti),
                        ]))
        elif BN == True and acti is None:
            return nn.Sequential(OrderedDict([
                            ( layer_name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)),
                            ('bn', nn.BatchNorm2d(out_channels)),
                        ]))
    elif layer_name in ["interpolate"]:
        return F.interpolate
        
            


class LayerTest(nn.Module):
    def __init__(self, config, debug=False):
        super().__init__()
        self.debug = debug
        layer_pars = config["model"]
        self.quantize = config["model"]["quantize"]
        all_layers = ["layer_00", "layer_01", "layer_02", "layer_03", "layer_04", "layer_05", "layer_06", "layer_07", "layer_08", "layer_09", "layer_10", "layer_11", "layer_12", "layer_13", "layer_14", "layer_15"]

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

                elif layer_config["type"] == "Conv":
                    in_channels = layer_config["in_channels"]
                    out_channels = layer_config["out_channels"]
                    groups = 1
                    layer_name = 'conv'

                elif layer_config["type"] == "Add_scalar":
                    layer_name = 'add_scalar'

                elif layer_config["type"] == "Interpolate":
                    in_channels = layer_config["in_channels"]
                    out_channels = layer_config["out_channels"]
                    layer_name = 'interpolate'

                
                kernel_size = (layer_config["kernel"], layer_config["kernel"])
                stride = layer_config["stride"]
                padding = (layer_config["padding"]["padding_w"], layer_config["padding"]["padding_h"])
                BN = layer_config["BN"]
                activation = layer_config["activation"]
                if activation:
                    if activation == "RELU6":
                        self.acti = nn.ReLU(inplace=False)
                    elif activation == "RELU":
                        self.acti = nn.ReLU(inplace=False)
                else:
                    self.acti = None

                if layer_name not in ["add_scalar"]:

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
                    elif id == 14:
                        if layer_config["set"] == True:
                            self.layer_14 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    elif id == 15:
                        if layer_config["set"] == True:
                            self.layer_15 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    
                if not activation:
                    if layer_name not in ["interpolate", "add_scalar"]:
                        if BN == True:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".bn"])
                        else:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name])

                else:
                    if layer_name not in ["interpolate", "add_scalar"]:
                        if BN == True:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".bn", "layer_"+str(id).zfill(2)+".relu"])
                        else:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".relu"])

        
        self.classification = nn.Linear(config["model"]["classification"]["in_channels"], config["training"]["classes"])   
        
        if self.quantize:
            self.quant = torch.quantization.QuantStub()
            self.quant_2 = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            self.add_1 = nn.quantized.FloatFunctional()
        self.xs = torch.tensor(1.0)
        self.scale_factor = (2.0, 2.0)
    def forward(self, x):
        

        activations = {}
        if self.quantize:
            x = self.quant(x)
        
        activations['input_image_tensor'] = x
        # DW
        x = self.layer_00(x)
        activations['OP_0_DEPTHWISECONV2D'] = x

        x = torch.reshape(x, (x.shape[0], 16, 30, 40)) 
        activations['OP_1_RESHAPE'] = x  

        # Conv2D, RELU
        x = self.layer_02(x)
        activations['OP_2_CONV2D'] = x  
        
        # Interpolate
        x = self.layer_03(x, scale_factor= self.scale_factor, mode='nearest')
        activations['OP_3_INTERPOLATE'] = x  
        
        # Conv2D, RELU
        x = self.layer_04(x)
        activations['OP_4_CONV2D'] = x  
        
        # Conv2D, RELU
        x = self.layer_05(x)
        activations['OP_5_CONV2D'] = x  
        # pdb.set_trace()
        # Interpolate
        x = self.layer_06(x, scale_factor=self.scale_factor,mode='nearest')    
        activations['OP_6_INTERPOLATE'] = x  
        
        # Conv2D
        x = self.layer_07(x)
        activations['OP_7_CONV2D'] = x  

        # x = torch.reshape(x, (x.shape[0], 10, 10, 8))
        # For debugging
        
        # x = torch.reshape(x, (x.shape[0], 10, 10, 8)).clone()

        x = torch.reshape(x, (x.shape[0], 10, 10, 8))
        activations['OP_8_RESHAPE'] = x

        # pdb.set_trace()
        # Interpolate
        x = self.layer_09(x, scale_factor=self.scale_factor, mode='nearest')
        activations['OP_9_INTERPOLATE'] = x  

        # Conv2D
        x = self.layer_10(x)
        activations['OP_10_CONV2D'] = x  

        x = torch.reshape(x, (x.shape[0], 48, 5, 5))
        activations['OP_11_RESHAPE'] = x  

        # Conv2D
        x = self.layer_12(x)
        activations['OP_12_CONV2D'] = x  
        
        # Interpolate
        x = self.layer_13(x, scale_factor=self.scale_factor, mode='nearest')
        activations['OP_13_INTERPOLATE'] = x  
        
        # Conv2D
        x = self.layer_14(x)
        activations['OP_14_CONV2D'] = x  

        x = self.dequant(x)
        x = torch.add(x, self.xs)
        
        x = self.quant_2(x)
        activations['OP_15_ADDSUB'] = x  
        
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        activations['OP_16_RESHAPE'] = x  
        
        x = self.classification(x)
        activations['OP_17_FULLYCONNECTED'] = x  

        x = self.dequant(x)

        x = F.softmax(x, dim=1)
        activations['OP_18_SOFTMAX'] = x  
        
                
        if self.debug:
            return x, activations
        else:
            return x



def build_model(config, debug):
    net = LayerTest(config, debug)
    return net

