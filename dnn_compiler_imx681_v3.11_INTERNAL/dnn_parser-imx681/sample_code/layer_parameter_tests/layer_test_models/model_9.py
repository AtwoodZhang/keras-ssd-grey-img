import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import pdb

def layer_declaration(layer_name="", in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=1, groups=1, acti="", BN=True, bias=False):
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
        elif BN == False and acti is None:
            return nn.Sequential(OrderedDict([
                            ( layer_name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)),
                        ]))
    elif layer_name in ["interpolate"]:
        return F.interpolate
    
    elif layer_name in ["relu"]:
        return nn.Sequential(OrderedDict([
                            ('relu', acti),
                        ]))
    elif layer_name in ["linear"]:
        if BN == True and acti is not None:
            return nn.Sequential(OrderedDict([
                                (layer_name, nn.Linear(in_channels, out_channels)),
                                ('bn', nn.BatchNorm2d(out_channels)),
                                ('relu', acti),
                            ]))
        elif BN == True and acti is None:
            return nn.Sequential(OrderedDict([
                                (layer_name, nn.Linear(in_channels, out_channels)),
                                ('bn', nn.BatchNorm2d(out_channels)),
                            ]))
        elif BN == False and acti is not None:
            return nn.Sequential(OrderedDict([
                                (layer_name, nn.Linear(in_channels, out_channels)),
                                ('relu', acti),
                            ]))
        elif BN == False and acti is  None:
            return nn.Sequential(OrderedDict([
                                (layer_name, nn.Linear(in_channels, out_channels)),
                            ]))
    
    elif layer_name in ["maxpool"]:
        return nn.Sequential(OrderedDict([
                                (layer_name, nn.MaxPool2d(kernel_size, stride, padding)),
                            ]))

    elif layer_name in ["sigmoid"]:
        return nn.Sigmoid()


class LayerTest(nn.Module):
    def __init__(self, config, debug=False):
        super().__init__()
        self.debug = debug
        layer_pars = config["model"]
        self.quantize = config["model"]["quantize"]
        all_layers = ["layer_00", "layer_01", "layer_02", "layer_03", "layer_04", "layer_05", "layer_06", "layer_07", "layer_08", "layer_09",  
                      "layer_10", "layer_11", "layer_12", "layer_13", "layer_14", "layer_15", "layer_16", "layer_17", "layer_18", "layer_19", 
                      "layer_20", "layer_21", "layer_22"]

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

                elif layer_config["type"] in  ["RELU", "RELU6"]:
                    layer_name = "relu"
                
                elif layer_config["type"] in ["FC", "Dense"]:
                    in_channels = layer_config["in_channels"]
                    out_channels = layer_config["out_channels"]
                    layer_name = "linear"
                    # pdb.set_trace()
                    # print("in_channels: ", in_channels)
                    # print("out_channels: ", out_channels)
                
                elif layer_config["type"] in ["MaxPool2D"]:
                    layer_name = "maxpool"
                
                elif layer_config["type"] in ["Sigmoid"]:
                    layer_name = "sigmoid"
                
                
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
                    elif id == 16:
                        if layer_config["set"] == True:
                            self.layer_16 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    elif id == 17:
                        if layer_config["set"] == True:
                            self.layer_17 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    elif id == 18:
                        if layer_config["set"] == True:
                            self.layer_18 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    elif id == 19:
                        if layer_config["set"] == True:
                            self.layer_19 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    elif id == 20:
                        if layer_config["set"] == True:
                            self.layer_20 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    elif id == 21:
                        if layer_config["set"] == True:
                            self.layer_21 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                    elif id == 22:
                        if layer_config["set"] == True:
                            self.layer_22 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
                if not activation:
                    if layer_name not in ["interpolate", "add_scalar", "maxpool", "linear", "sigmoid"]: #linear can be fused only with relu (linear, relu)
                        if BN == True:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".bn"])
                        else:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name])

                else:
                    if layer_name not in ["interpolate", "add_scalar", "relu", "maxpool", "sigmoid"]:
                        if BN == True:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".bn", "layer_"+str(id).zfill(2)+".relu"])
                        else:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".relu"])
                    # elif layer_name == "relu":
                    #     if BN == True:
                    #         self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+".bn", "layer_"+str(id).zfill(2)+"."+layer_name])
                    #     else:
                    #         self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name])
        
        # pdb.set_trace()
        if config["model"]["classification"]["set"] == True: 
            self.classification = nn.Linear(config["model"]["classification"]["in_channels"], config["training"]["classes"])   
        if config["model"]["sigmoid"]["set"] == True: 
            self.sigmoid = nn.Sigmoid()
        if self.quantize:
            self.quant = torch.quantization.QuantStub()
            self.quant_2 = torch.quantization.QuantStub()
            self.quant_3 = torch.quantization.QuantStub()
            self.quant_4 = torch.quantization.QuantStub()
            self.quant_5 = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            self.concat_1 = nn.quantized.FloatFunctional()
            self.concat_2 = nn.quantized.FloatFunctional()
            self.concat_3 = nn.quantized.FloatFunctional()
            self.add_1 = nn.quantized.FloatFunctional()
            self.sub_1 = nn.quantized.FloatFunctional()
            self.mul_1 = nn.quantized.FloatFunctional()
        self.xs = torch.tensor(0.2)
        self.xs_2 = torch.tensor(3.0)     
        self.xs_3 = torch.tensor(0.5)     
    def forward(self, x):

        ## Order of execution

        
        activations = {}

        if self.quantize:
            x = self.quant(x)
        
        activations['input_image_tensor'] = x
        
        
        x = torch.reshape(x,(8,24,10,10))
        activations['OP_0_RESHAPE'] = x
        
        # Conv2D
        x1 = self.layer_01(x)
        activations['OP_1_CONV2D'] = x1
        
        # DW
        x2 = self.layer_02(x1)
        activations['OP_2_DEPTHWISECONV2D'] = x2
        
        # concat
        x =self.concat_1.cat((x1, x2), dim=2)
        activations['OP_3_CONCATENATE'] = x
        
        # Relu
        x3 = self.layer_04(x)
        activations['OP_4_RELU'] = x3
        
        # DW
        x4 = self.layer_05(x3)
        activations['OP_5_DEPTHWISECONV2D'] = x4
        
        # concat
        x = self.concat_1.cat((x3, x4), dim=3)
        activations['OP_6_CONCATENATE'] = x
        
        # MaxPool
        
        x5 = self.layer_07(x)
        activations['OP_7_MAXPOOL'] = x5
        
        # conv
        x6 = self.layer_08(x5)
        activations['OP_8_CONV2D'] = x6
        
        # x5 = self.dequant(x5)
        # x6 = self.dequant(x6)

        x = self.mul_1.mul(x5, x6)
        activations['OP_9_MULTIPLY'] = x
        
        # Sigmoid
        x= self.layer_10(x)
        activations['OP_10_SIGMOID'] = x
        
        x = self.dequant(x)
    
        x = torch.mul(x, self.xs)
        
        
        x = self.quant_2(x)
        activations['OP_11_MULTIPLY'] = x
        
        x = torch.reshape(x,(8, 1200))
        activations['OP_12_RESHAPE'] = x
        

        x = self.layer_13(x)
        activations['OP_13_FULLYCONNECTED'] = x
        

        x = torch.reshape(x,(1, 80))
        activations['OP_14_RESHAPE'] = x

        x = self.layer_15(x)
        activations['OP_15_FULLYCONNECTED'] = x
        

        x = self.dequant(x)
        
        # Softmax
        x = F.softmax(x, dim=1)

        # x_softmax = self.quant_3(x)
        # activations['OP_16_SOFTMAX'] = x_softmax
        activations['OP_16_SOFTMAX'] = x
        
        

        if self.debug:
            return x, activations
        else:
            return x



def build_model(config, debug):
    net = LayerTest(config, debug)
    return net

