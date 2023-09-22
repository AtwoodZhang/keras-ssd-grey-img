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
            # return nn.Linear(in_channels, out_channels)
                    
    
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
                      "layer_20", "layer_21", "layer_22", "layer_23"]

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
                    elif id == 23:
                        if layer_config["set"] == True:
                            self.layer_23 = layer_declaration(layer_name, in_channels, out_channels, kernel_size, stride, padding, groups, self.acti, BN, bias=False)
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
            self.dequant = torch.quantization.DeQuantStub()
            self.concat_1 = nn.quantized.FloatFunctional()
            self.concat_2 = nn.quantized.FloatFunctional()
            self.concat_3 = nn.quantized.FloatFunctional()
            self.add_1 = nn.quantized.FloatFunctional()
            
    def forward(self, x):

        ## Order of execution

        activations = {}
        if self.quantize:
            x = self.quant(x)

        activations['input_image_tensor'] = x
        

        # Depthwise Conv2D
        x = self.layer_00(x) 
        activations['OP_0_DEPTHWISECONV2D'] = x
        # Reshape
        x_re_00 = torch.reshape(x, (x.shape[0], 4, 30, 40)) 
        activations['OP_1_RESHAPE'] = x_re_00

        # Conv, RELU6
        x2 = self.layer_02(x_re_00)
        activations['OP_2_CONV2D'] = x2

        # Reshape
        x_re_02 = torch.reshape(x2, (x2.shape[0], 4, 30, 10)) 
        activations['OP_3_RESHAPE'] = x_re_02
        

        # Conv, RELU6
        x4 = self.layer_04(x_re_00)
        activations['OP_4_CONV2D'] = x4
        # Reshape
        x_re_04 = torch.reshape(x4, (x4.shape[0], 4, 30, 15)) 
        activations['OP_5_RESHAPE'] = x_re_04
        

        # Concatenate x_re_00, x_re_02, x_re_04
        if self.quantize:
            x_06_concat_1 = self.concat_1.cat((x_re_00, x_re_02, x_re_04), dim=-1)
        else:
            x_06_concat_1 = torch.cat((x_re_00, x_re_02, x_re_04), dim=-1)

        activations['OP_6_CONCATENATE'] = x_06_concat_1
        # Conv, RELU6
        x_07 = self.layer_07(x_06_concat_1)
        activations['OP_7_CONV2D'] = x_07
        # Reshape
        x_re_07 = torch.reshape(x_07, (x_07.shape[0], 75, 11, 3)) 
        activations['OP_8_RESHAPE'] = x_re_07

        # Conv, RELU6
        x_09 = self.layer_09(x_06_concat_1)
        activations['OP_9_CONV2D'] = x_09
        # Reshape
        x_re_09 = torch.reshape(x_09, (x_09.shape[0], 75, 22, 3)) 
        activations['OP_10_RESHAPE'] = x_re_09

        
        # Conv, RELU6
        x_11 = self.layer_11(x_06_concat_1)
        activations['OP_11_CONV2D'] = x_11
        # Reshape
        x_re_11 = torch.reshape(x_11, (x_11.shape[0], 75, 33, 3)) 
        activations['OP_12_RESHAPE'] = x_re_11
        
        
        # Concatenate x_re_07, x_re_09, x_re_11
        if self.quantize:
            x_13_concat_2 = self.concat_2.cat((x_re_07, x_re_09, x_re_11), dim=2)
        else:
            x_13_concat_2 = torch.cat((x_re_07, x_re_09, x_re_11), dim=2)
        
        activations['OP_13_CONCATENATE'] = x_13_concat_2
        # Reshape 
        x_re_concat_02 = torch.reshape(x_13_concat_2, (x_13_concat_2.shape[0], 90, 11, 15)) 
        activations['OP_14_RESHAPE'] = x_re_concat_02

        # Conv, RELU6
        x_15 = self.layer_15(x_re_concat_02)
        activations['OP_15_CONV2D'] = x_15
        x_re_15 = torch.reshape(x_15, (x_15.shape[0], 66, 5, 5)) 
        activations['OP_16_RESHAPE'] = x_re_15


        # Conv, RELU6
        x_17 = self.layer_17(x_re_concat_02)
        activations['OP_17_CONV2D'] = x_17
        # Reshape
        x_re_17 = torch.reshape(x_17, (x_17.shape[0], 33, 5, 5)) 
        activations['OP_18_RESHAPE'] = x_re_17
        
        
        # Conv, RELU6
        x_19 = self.layer_19(x_re_concat_02)
        activations['OP_19_CONV2D'] = x_19
        # Reshape
        x_re_19 = torch.reshape(x_19, (x_19.shape[0], 99, 5, 5)) 
        activations['OP_20_RESHAPE'] = x_re_19
    

        # Concatenate x_re_15, x_re_17, x_re_19
        if self.quantize:
            x_21_concat_3 = self.concat_3.cat((x_re_15, x_re_17, x_re_19), dim=1)
        else:    
            x_21_concat_3 = torch.cat((x_re_15, x_re_17, x_re_19), dim=1)
        
        activations['OP_21_CONCATENATE'] = x_21_concat_3

        # Conv, RELU6
        x_22 = self.layer_22(x_21_concat_3)        
        activations['OP_22_CONV2D'] = x_22

        # x = x_22.view(x_22.shape[0], x_22.shape[1] * x_22.shape[2]*x_22.shape[3])
        x = torch.reshape(x_22, (x_22.shape[0], x_22.shape[1] * x_22.shape[2] * x_22.shape[3]))
        activations['OP_23_RESHAPE'] = x
        # x = self.classification(x)

        
        #Dense (FC)
        x = self.layer_23(x)
        activations['OP_24_FULLYCONNECTED'] = x

        x = self.dequant(x)

        x = F.softmax(x, dim=1)
        
        activations['OP_25_SOFTMAX'] = x

        if self.debug:
            return x, activations
        else:
            return x



def build_model(config, debug):
    # pdb.set_trace()
    net = LayerTest(config, debug)
    return net

