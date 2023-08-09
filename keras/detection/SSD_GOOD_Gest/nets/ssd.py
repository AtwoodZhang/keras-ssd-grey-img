import keras.backend as K
import numpy as np
from keras.layers import (Activation, Concatenate, Conv2D, Flatten, Input, Reshape, DepthwiseConv2D)
from keras.models import Model
from nets.mobilenet import mobilenet


# from mobilenet import mobilenet

# from mobilenet import mobilenet

def SSD300(input_shape, num_classes=2):
    input_tensor = Input(shape=input_shape)  # 输入为：[120, 160, 1]

    # step1. 提取主干特征；
    net = mobilenet(input_tensor)

    # ---------------------------将提取到的主干特征进行处理--------------------------#
    # 对net['split_layer1']的通道进行l2标准化处理
    # 15, 20, 64
    num_prior = 1
    # 1) layer1-cls-confidence
    net['split_layer1_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                   use_bias=True, activation="relu", name='DepthwiseConv2D_conf_DD1_1')(
        net['split_layer1'])
    net['split_layer1_conf_Conv'] = Conv2D(num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                           name='Conv2D_conf_DD1_2')(net['split_layer1_conf_Dep'])
    net['split_layer1_conf_Reshape'] = Reshape((600, 2))(net['split_layer1_conf_Conv'])
    #    layer1-bbox, 4是x, y, h, w的调整
    net['split_layer1_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                  use_bias=True, activation='relu', name='DepthwiseConv2D_loc_DD1_1')(
        net['split_layer1'])
    net['split_layer1_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                          name='Conv2D_loc_DD1_2')(net['split_layer1_loc_Dep'])
    net['split_layer1_loc_Reshape'] = Reshape((600, 1, 4))(net['split_layer1_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer2']的通道进行处理
    # 8 * 10 * 80
    num_prior = 3
    # 2) layer2-cls-confidence
    net['split_layer2_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                   use_bias=True, activation='relu', name='DepthwiseConv2D_conf_DD2_1')(
        net['split_layer2'])
    net['split_layer2_conf_Conv'] = Conv2D(num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                           name='Conv2D_conf_DD2_2')(net['split_layer2_conf_Dep'])
    net['split_layer2_conf_Reshape'] = Reshape((480, 2))(net['split_layer2_conf_Conv'])
    #    layer2-bbox, 4是x, y, h, w的调整
    net['split_layer2_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                  use_bias=True, activation='relu', name='DepthwiseConv2D_loc_DD2_1')(
        net['split_layer2'])
    net['split_layer2_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                          name='Conv2D_loc_DD2_2')(net['split_layer2_loc_Dep'])
    net['split_layer2_loc_Reshape'] = Reshape((480, 1, 4))(net['split_layer2_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer3']的通道进行处理
    # 4, 5, 80
    num_prior = 3
    # 3) layer3-cls-confidence
    net['split_layer3_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                   use_bias=True, activation='relu', name='DepthwiseConv2D_conf_DD3_1')(
        net['split_layer3'])
    net['split_layer3_conf_Conv'] = Conv2D(num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                           name='Conv2D_conf_DD3_2')(net['split_layer3_conf_Dep'])
    net['split_layer3_conf_Reshape'] = Reshape((120, 2))(net['split_layer3_conf_Conv'])
    #    layer3-bbox, 4是x, y, h, w的调整
    net['split_layer3_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                  use_bias=True, activation='relu', name='DepthwiseConv2D_loc_DD3_1')(
        net['split_layer3'])
    net['split_layer3_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                          name='Conv2D_loc_DD3_2')(net['split_layer3_loc_Dep'])
    net['split_layer3_loc_Reshape'] = Reshape((120, 1, 4))(net['split_layer3_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer4']的通道进行处理
    # 2, 3, 64
    num_prior = 3
    # 4) layer4-cls-confidence
    net['split_layer4_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                   use_bias=True, activation='relu', name='DepthwiseConv2D_conf_DD4_1')(
        net['split_layer4'])
    net['split_layer4_conf_Conv'] = Conv2D(num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                           name='Conv2D_conf_DD4_2')(net['split_layer4_conf_Dep'])
    net['split_layer4_conf_Reshape'] = Reshape((36, 2))(net['split_layer4_conf_Conv'])
    #    layer4-bbox, 4是x, y, h, w的调整
    net['split_layer4_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                  use_bias=True, activation='relu', name='DepthwiseConv2D_loc_DD4_1')(
        net['split_layer4'])
    net['split_layer4_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                          name='Conv2D_loc_DD4_2')(net['split_layer4_loc_Dep'])
    net['split_layer4_loc_Reshape'] = Reshape((36, 1, 4))(net['split_layer4_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer5']的通道进行处理
    # 1, 1, 64
    num_prior = 3
    # 5) layer5-cls-confidence
    net['split_layer5_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                   use_bias=True, activation='relu', name='DepthwiseConv2D_conf_DD5_1')(
        net['split_layer5'])
    net['split_layer5_conf_Conv'] = Conv2D(num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                           name='Conv2D_conf_DD5_2')(net['split_layer5_conf_Dep'])
    net['split_layer5_conf_Reshape'] = Reshape((6, 2))(net['split_layer5_conf_Conv'])
    #    layer5-bbox, 4是x, y, h, w的调整
    net['split_layer5_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1),
                                                  use_bias=True, activation='relu', name='DepthwiseConv2D_loc_DD5_1')(
        net['split_layer5'])
    net['split_layer5_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),
                                          name='Conv2D_loc_DD5_2')(net['split_layer5_loc_Dep'])
    net['split_layer5_loc_Reshape'] = Reshape((6, 1, 4))(net['split_layer5_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 将所有结果进行堆叠
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['split_layer1_loc_Reshape'],
                                                            net['split_layer2_loc_Reshape'],
                                                            net['split_layer3_loc_Reshape'],
                                                            net['split_layer4_loc_Reshape'],
                                                            net['split_layer5_loc_Reshape']])

    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['split_layer1_conf_Reshape'],
                                                              net['split_layer2_conf_Reshape'],
                                                              net['split_layer3_conf_Reshape'],
                                                              net['split_layer4_conf_Reshape'],
                                                              net['split_layer5_conf_Reshape']])

    # 1242,4
    net['mbox_loc'] = Reshape((1242, 4), name='mbox_loc_final')(net['mbox_loc'])
    # 1242,2
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])

    # # model_return
    # net_output = []
    # net_output.append(net['mbox_loc'])
    # net_output.append(net['mbox_conf'])
    # # net_output.append(net['mbox_loc'])
    # print(net_output)
    # model = Model(input_tensor, net_output)

    net['predictions'] = Concatenate(axis=-1, name='predictions')([net['mbox_loc'], net['mbox_conf']])
    model = Model(input_tensor, net['predictions'])
    # print(K.int_shape(net_output))
    # model.summary()
    return model


if __name__ == "__main__":
    model = SSD300([120, 160, 3])
    # model.save(
    #     "/home/zya/zya/AI/NNet/detection/class_01_mobilenet_ssd/test1_from_bubbliiing/model_data/model_structure_test.h5")
    # print("Save_end")
    model.summary()
