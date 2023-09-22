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
                    if layer_name not in ["interpolate", "add_scalar", "maxpool", "linear"]: #linear can be fused only with relu (linear, relu)
                        if BN == True:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name, "layer_"+str(id).zfill(2)+".bn"])
                        else:
                            self.all_layer_conv_names.append(["layer_"+str(id).zfill(2)+"."+layer_name])

                else:
                    if layer_name not in ["interpolate", "add_scalar", "relu", "maxpool"]:
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
            self.dequant = torch.quantization.DeQuantStub()
            self.concat_1 = nn.quantized.FloatFunctional()
            self.concat_2 = nn.quantized.FloatFunctional()
            self.concat_3 = nn.quantized.FloatFunctional()
            self.add_1 = nn.quantized.FloatFunctional()

        self.xs = torch.tensor(0.5)
        # self.add_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # self.add_values = torch.Tensor(self.add_values)
        # self.add_values = torch.reshape(self.add_values, (1,10,1,1)).cuda()

    def forward(self, x):

        ## Order of execution

        activations = {}

        if self.quantize:
            x = self.quant(x)
        
        activations['input_image_tensor'] = x

        # Depthwise Conv2D
        x = self.layer_00(x) 
        activations['OP_0_DEPTHWISECONV2D'] = x

        # Depthwise Conv2D
        x = self.layer_01(x)
        activations['OP_1_DEPTHWISECONV2D'] = x

        # Depthwise Conv2D
        x = self.layer_02(x)
        activations['OP_2_DEPTHWISECONV2D'] = x

        # Depthwise Conv2D
        x = self.layer_03(x)
        activations['OP_3_DEPTHWISECONV2D'] = x

        # Depthwise Conv2D
        x = self.layer_04(x)
        activations['OP_4_DEPTHWISECONV2D'] = x

        # Maxpool
        x = self.layer_05(x)
        activations['OP_5_MAXPOOL'] = x

        # Conv2D
        x = self.layer_06(x)
        activations['OP_6_CONV2D'] = x

        # Conv2D
        x = self.layer_07(x)
        activations['OP_7_CONV2D'] = x



        # COnv2D
        x = self.layer_08(x)
        activations['OP_8_CONV2D'] = x

        # Conv2D, RELU
        x = self.layer_09(x)
        activations['OP_9_CONV2D'] = x
        

        # Maxpool
        x = self.layer_10(x)
        activations['OP_10_MAXPOOL'] = x

        # Conv2D
        x = self.layer_11(x)
        activations['OP_11_CONV2D'] = x
        
        
        x = torch.reshape(x,(x.shape[0], 96, 80) )
        activations['OP_12_RESHAPE'] = x

        x = torch.reshape(x,(x.shape[0], 40, 12, 16) )
        activations['OP_13_RESHAPE'] = x
        
        # Maxpool
        x = self.layer_13(x)
        activations['OP_14_MAXPOOL'] = x
        

        x = self.dequant(x)
        x = torch.add(x, self.xs)
        
        x = self.quant_2(x)
        activations['OP_15_ADDSUB'] = x


        # Conv2D, RELU
        x = self.layer_15(x)
        activations['OP_16_CONV2D'] = x
        

        # Conv2D
        x = self.layer_16(x)
        activations['OP_17_CONV2D'] = x
        

        x = self.dequant(x)

        # x = torch.add(x, self.add_values)

        # Softmax
        x = F.softmax(x, dim=1)
        

        x = self.quant_3(x)

        activations['OP_18_SOFTMAX'] = x

        x = torch.reshape(x,(x.shape[0], 120) )
        activations['OP_19_RESHAPE'] = x
        
        

        # Dense (FC)
        x = self.layer_20(x)
        activations['OP_20_FULLYCONNECTED'] = x
        
        x = self.dequant(x)
        
        # Softmax
        x = F.softmax(x, dim=1)

        # x_softmax_quant = self.quant_4(x)
        # activations['OP_21_SOFTMAX'] = x_softmax_quant
        activations['OP_21_SOFTMAX'] = x
        
        if self.debug:
            return x, activations
        else:
            return x



def build_model(config, debug):
    net = LayerTest(config, debug)
    return net

