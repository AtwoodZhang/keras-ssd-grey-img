import tensorflow as tf
from keras.layers import (Conv2D, Dense, DepthwiseConv2D, add, BatchNormalization)
from keras.optimizers import SGD, Adam
import keras.backend as K
import numpy as np
import math 
import keras
from PIL import Image
from random import shuffle
from keras import layers as KL
from keras.regularizers import l2
from keras.layers import (Activation, Concatenate, Conv2D, Flatten, Input, Reshape, DepthwiseConv2D)
from keras.models import Model
from keras import layers as KL

# from mobilenet import mobilenet


### 1. 定义 MobileNet 网络
# block of DepthwiseConv2D and Conv2D
def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=strides, use_bias=True,name='block_%d_conv_dw' % block_id)(inputs)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    x = Conv2D(pointwise_conv_filters, kernel_size=(1, 1), padding="same", use_bias=True, strides=strides,name='block_%d_conv_pw' % block_id)(x)
    x = add([inputs, x])
    return x

def _depthwise_conv_block_no_relu(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=strides, use_bias=True,name='block_%d_conv_dw' % block_id)(inputs)
    x = Conv2D(pointwise_conv_filters, kernel_size=(1, 1), padding="same", use_bias=True, strides=strides,name='block_%d_conv_pw' % block_id)(x)
    x = add([inputs, x])
    return x

### 1) block of backbone
# block of backbone
def mobilenet(inputs_tensor):
    # --------------------------主干特征提取网络开始--------------------------#
    # SSD结构，net字典
    net = {}
    # inputs_tensor: 120 * 160 * 1 --> 120 * 160 * 8
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=8, strides=(1, 1), use_bias=True, name='DepthWiseConv2D_layer1')(inputs_tensor)
    # x = features
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    net['first_layer'] = x
    # --> 60 * 80 * 8 
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(2, 2), use_bias=True, name='DepthWiseConv2D_layer2')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 60 * 80 * 8
    x = Conv2D(8, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer3')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 30 * 40 * 16                               
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2, strides=(2, 2), use_bias=True, name='DepthWiseConv2D_layer4')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 30 * 40 * 8  
    x = Conv2D(8, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer5')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2

    # Block 1~3  
    x = _depthwise_conv_block(x, 8, 1, block_id=1)  # --> 30 * 40 * 8
    x = _depthwise_conv_block(x, 8, 1, block_id=2)  # --> 30 * 40 * 8
    x = _depthwise_conv_block(x, 8, 1, block_id=3)  # --> 30 * 40 * 8             

    # Conv_Depth_Conv_Depth_Conv
    # --> 30 * 40 * 16
    x = Conv2D(16, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer6')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 30 * 40 * 16
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthWiseConv2D_layer7')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 30 * 40 * 16
    x = Conv2D(16, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer8')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 15 * 20 * 32
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2, strides=(2, 2), use_bias=True, name='DepthWiseConv2D_layer9')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 15 * 20 * 24 
    x = Conv2D(24, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer10')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2

    # Block 4~6
    x = _depthwise_conv_block(x, 24, 1, block_id=4)  # --> 15 * 20 * 24
    x = _depthwise_conv_block(x, 24, 1, block_id=5)  # --> 15 * 20 * 24
    x = _depthwise_conv_block(x, 24, 1, block_id=6)  # --> 15 * 20 * 24

    # --> 15 * 20 * 48
    x = Conv2D(48, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer11')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 15 * 20 * 48
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthWiseConv2D_layer12')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 15 * 20 * 64
    x = Conv2D(64, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer13', kernel_regularizer='l2')(x)
    # x = Conv2D(64, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer13')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    net['split_layer1'] = x

    # start split;
    # --> 8 * 10 * 64
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthWiseConv2D_layer14')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 8 * 10 * 40
    x = Conv2D(40, (1, 1), padding='same', use_bias=True, strides=(2, 2), name='Conv2D_layer15')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # block 7_1 ~7_2
    x = _depthwise_conv_block(x, 40, 1, block_id=7)  # --> 8 * 10 * 40
    x = _depthwise_conv_block(x, 40, 1, block_id=8)  # --> 8 * 10 * 40
    # --> 8 * 10 * 80
    x = Conv2D(80, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer16')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 8 * 10 * 80
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthWiseConv2D_layer17')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 8 * 10 * 80
    x = Conv2D(80, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer18')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    net['split_layer2'] = x

    # --> 4 * 5 * 80
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(2, 2), use_bias=True, name='DepthWiseConv2D_layer19')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 4 * 5 * 80
    x = Conv2D(80, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer20')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    net['split_layer3'] = x

    # --> 2 * 3 * 80
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(2, 2), use_bias=True, name='DepthWiseConv2D_layer21')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    # --> 2 * 3 * 64
    x = Conv2D(64, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer22')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    net['split_layer4'] = x

    # --> 1 * 1 * 64
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(3, 3), use_bias=True, name='DepthWiseConv2D_layer23')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    x = Conv2D(64, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer24')(x)
    try:
        x = KL.ReLU(max_value=6.)(x)
    except:
        x = tf.nn.relu(x)  # tf 1.13.2
    net['split_layer5'] = x

    # -------------------------------------主干特征提取网络结束--------------------------------#
    return net


def SSD300_bk(input_shape, num_classes=2, features=''):

    input_tensor = Input(shape=input_shape)  # 输入为：[120, 160, 1]

    # step1. 提取主干特征；
    net = mobilenet(input_tensor, features)
    # model = Model(input_tensor, [net['split_layer1'], net['split_layer2'], net['split_layer3'], net['split_layer4'],net['split_layer5']])
    # net['split_layer1'] = features[4]
    # net['split_layer2'] = features[3]
    # net['split_layer3'] = features[2]
    # net['split_layer4'] = features[1]
    # net['split_layer5'] = features[0]
    num_prior = 1
    # 1) layer1-cls-confidence
    net['split_layer1_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD1_1')(net['split_layer1'])
    try:
        net['split_layer1_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer1_conf_Dep'])
    except:
        net['split_layer1_conf_Dep'] = tf.nn.relu(net['split_layer1_conf_Dep'])
    net['split_layer1_conf_Conv'] = Conv2D(num_classes * 2 * num_prior, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD1_2')(net['split_layer1_conf_Dep'])
    # net['split_layer1_conf_Reshape'] = Reshape((600, 2))(net['split_layer1_conf_Conv'])
    net['split_layer1_conf_Reshape'] = Reshape((600, num_classes))(net['split_layer1_conf_Conv'])
    #    layer1-bbox, 4是x, y, h, w的调整
    net['split_layer1_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD1_1')(net['split_layer1'])
    try:
        net['split_layer1_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer1_loc_Dep'])
    except:
        net['split_layer1_loc_Dep'] = tf.nn.relu(net['split_layer1_loc_Dep'])
    net['split_layer1_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD1_2')(net['split_layer1_loc_Dep'])
    # net['split_layer1_loc_Reshape'] = Reshape((600, 1, 4))(net['split_layer1_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer1_loc_Reshape'] = Reshape((600, 4))(net['split_layer1_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer2']的通道进行处理
    # 8 * 10 * 80
    num_prior = 3
    
    
    # 2) layer2-cls-confidence
    net['split_layer2_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD2_1')(net['split_layer2'])
    try:
        net['split_layer2_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer2_conf_Dep'])
    except:
        net['split_layer2_conf_Dep'] = tf.nn.relu(net['split_layer2_conf_Dep'])
    net['split_layer2_conf_Conv'] = Conv2D(num_classes * 2 * num_prior, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD2_2')(net['split_layer2_conf_Dep'])
    # net['split_layer2_conf_Reshape'] = Reshape((480, 2))(net['split_layer2_conf_Conv'])
    net['split_layer2_conf_Reshape'] = Reshape((480, num_classes))(net['split_layer2_conf_Conv'])
    #    layer2-bbox, 4是x, y, h, w的调整
    net['split_layer2_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD2_1')(net['split_layer2'])
    try:
        net['split_layer2_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer2_loc_Dep'])
    except:
        net['split_layer2_loc_Dep'] = tf.nn.relu(net['split_layer2_loc_Dep'])
    net['split_layer2_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD2_2')(net['split_layer2_loc_Dep'])
    # net['split_layer2_loc_Reshape'] = Reshape((480, 1, 4))(net['split_layer2_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer2_loc_Reshape'] = Reshape((480, 4))(net['split_layer2_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer3']的通道进行处理
    # 4, 5, 80
    num_prior = 3
    
    
    # 3) layer3-cls-confidence
    net['split_layer3_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD3_1')(net['split_layer3'])
    try:
        net['split_layer3_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer3_conf_Dep'])
    except:
        net['split_layer3_conf_Dep'] = tf.nn.relu(net['split_layer3_conf_Dep'])
    net['split_layer3_conf_Conv'] = Conv2D(num_prior * 2 * num_classes, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD3_2')(net['split_layer3_conf_Dep'])
    # net['split_layer3_conf_Reshape'] = Reshape((120, 2))(net['split_layer3_conf_Conv'])
    net['split_layer3_conf_Reshape'] = Reshape((120, num_classes))(net['split_layer3_conf_Conv'])
    #    layer3-bbox, 4是x, y, h, w的调整
    net['split_layer3_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD3_1')(net['split_layer3'])
    try:
        net['split_layer3_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer3_loc_Dep'])
    except:
        net['split_layer3_loc_Dep'] = tf.nn.relu(net['split_layer3_loc_Dep'])
    net['split_layer3_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD3_2')(net['split_layer3_loc_Dep'])
    # net['split_layer3_loc_Reshape'] = Reshape((120, 1, 4))(net['split_layer3_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer3_loc_Reshape'] = Reshape((120, 4))(net['split_layer3_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer4']的通道进行处理
    # 2, 3, 64
    num_prior = 3
    
    
    # 4) layer4-cls-confidence
    net['split_layer4_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD4_1')(net['split_layer4'])
    try:
        net['split_layer4_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer4_conf_Dep'])
    except:
        net['split_layer4_conf_Dep'] = tf.nn.relu(net['split_layer4_conf_Dep'])
    net['split_layer4_conf_Conv'] = Conv2D(num_prior * 2* num_classes, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD4_2')(net['split_layer4_conf_Dep'])
    # net['split_layer4_conf_Reshape'] = Reshape((36, 2))(net['split_layer4_conf_Conv'])
    net['split_layer4_conf_Reshape'] = Reshape((36, num_classes))(net['split_layer4_conf_Conv'])
    #    layer4-bbox, 4是x, y, h, w的调整
    net['split_layer4_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD4_1')(net['split_layer4'])
    try:
        net['split_layer4_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer4_loc_Dep'])
    except:
        net['split_layer4_loc_Dep'] = tf.nn.relu(net['split_layer4_loc_Dep'])
    net['split_layer4_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD4_2')(net['split_layer4_loc_Dep'])
    # net['split_layer4_loc_Reshape'] = Reshape((36, 1, 4))(net['split_layer4_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer4_loc_Reshape'] = Reshape((36, 4))(net['split_layer4_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer5']的通道进行处理
    # 1, 1, 64
    num_prior = 3
    
    
    # 5) layer5-cls-confidence
    net['split_layer5_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD5_1')(net['split_layer5'])
    try:
        net['split_layer5_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer5_conf_Dep'])
    except:
        net['split_layer5_conf_Dep'] = tf.nn.relu(net['split_layer5_conf_Dep'])
    net['split_layer5_conf_Conv'] = Conv2D(num_prior * 2 * num_classes, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD5_2')(net['split_layer5_conf_Dep'])
    # net['split_layer5_conf_Reshape'] = Reshape((6, 2))(net['split_layer5_conf_Conv'])
    net['split_layer5_conf_Reshape'] = Reshape((6, num_classes))(net['split_layer5_conf_Conv'])
    
    #    layer5-bbox, 4是x, y, h, w的调整
    net['split_layer5_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD5_1')(net['split_layer5'])
    try:
        net['split_layer5_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer5_loc_Dep'])
    except:
        net['split_layer5_loc_Dep'] = tf.nn.relu(net['split_layer5_loc_Dep'])
    net['split_layer5_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),name='Conv2D_loc_DD5_2')(net['split_layer5_loc_Dep'])
    # net['split_layer5_loc_Reshape'] = Reshape((6, 1, 4))(net['split_layer5_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer5_loc_Reshape'] = Reshape((6, 4))(net['split_layer5_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 将所有结果进行堆叠
    net['cls_conf'] = Concatenate(axis=1, name='mbox_conf')([net['split_layer1_conf_Reshape'],
                                                              net['split_layer2_conf_Reshape'],
                                                              net['split_layer3_conf_Reshape'],
                                                              net['split_layer4_conf_Reshape'],
                                                              net['split_layer5_conf_Reshape']])
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['split_layer1_loc_Reshape'],
                                                            net['split_layer2_loc_Reshape'],
                                                            net['split_layer3_loc_Reshape'],
                                                            net['split_layer4_loc_Reshape'],
                                                            net['split_layer5_loc_Reshape']])

    # 1242,2
    net['cls_conf'] = Activation('softmax', name='cls_conf_final')(net['cls_conf'])
    # 1242,4
    net['mbox_loc'] = Reshape((1242, 4), name='mbox_loc_final')(net['mbox_loc'])

#分别训练
    # model_return
    # net_output = []
    # net_output.append(net['mbox_loc'])
    # net_output.append(net['cls_conf'])

    # model = Model(input_tensor, net_output)
    
# 一起训练
    # 1242,6
    net['predictions']  = Concatenate(axis =-1, name='predictions')([net['mbox_loc'], net['cls_conf']])

    
    # model = Model(input_tensor, net['predictions'])
    model = Model(input_tensor, net['first_layer'])
    
    return model

### 2) block of SSD head
def SSD300(input_shape, num_classes=2):

    input_tensor = Input(shape=input_shape)  # 输入为：[120, 160, 1]

    # step1. 提取主干特征；
    net = mobilenet(input_tensor)

    # ---------------------------将提取到的主干特征进行处理--------------------------#
    # 对net['split_layer1']的通道进行l2标准化处理
    # 15, 20, 64
    num_prior = 1
    # 1) layer1-cls-confidence
    net['split_layer1_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD1_1')(net['split_layer1'])
    try:
        net['split_layer1_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer1_conf_Dep'])
    except:
        net['split_layer1_conf_Dep'] = tf.nn.relu(net['split_layer1_conf_Dep'])
    net['split_layer1_conf_Conv'] = Conv2D(num_classes * 2 * num_prior, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD1_2')(net['split_layer1_conf_Dep'])
    # net['split_layer1_conf_Reshape'] = Reshape((600, 2))(net['split_layer1_conf_Conv'])
    net['split_layer1_conf_Reshape'] = Reshape((600, num_classes))(net['split_layer1_conf_Conv'])
    #    layer1-bbox, 4是x, y, h, w的调整
    net['split_layer1_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD1_1')(net['split_layer1'])
    try:
        net['split_layer1_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer1_loc_Dep'])
    except:
        net['split_layer1_loc_Dep'] = tf.nn.relu(net['split_layer1_loc_Dep'])
    net['split_layer1_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD1_2')(net['split_layer1_loc_Dep'])
    # net['split_layer1_loc_Reshape'] = Reshape((600, 1, 4))(net['split_layer1_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer1_loc_Reshape'] = Reshape((600, 4))(net['split_layer1_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer2']的通道进行处理
    # 8 * 10 * 80
    num_prior = 3
    
    
    # 2) layer2-cls-confidence
    net['split_layer2_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD2_1')(net['split_layer2'])
    try:
        net['split_layer2_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer2_conf_Dep'])
    except:
        net['split_layer2_conf_Dep'] = tf.nn.relu(net['split_layer2_conf_Dep'])
    net['split_layer2_conf_Conv'] = Conv2D(num_classes * 2 * num_prior, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD2_2')(net['split_layer2_conf_Dep'])
    # net['split_layer2_conf_Reshape'] = Reshape((480, 2))(net['split_layer2_conf_Conv'])
    net['split_layer2_conf_Reshape'] = Reshape((480, num_classes))(net['split_layer2_conf_Conv'])
    #    layer2-bbox, 4是x, y, h, w的调整
    net['split_layer2_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD2_1')(net['split_layer2'])
    try:
        net['split_layer2_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer2_loc_Dep'])
    except:
        net['split_layer2_loc_Dep'] = tf.nn.relu(net['split_layer2_loc_Dep'])
    net['split_layer2_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD2_2')(net['split_layer2_loc_Dep'])
    # net['split_layer2_loc_Reshape'] = Reshape((480, 1, 4))(net['split_layer2_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer2_loc_Reshape'] = Reshape((480, 4))(net['split_layer2_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer3']的通道进行处理
    # 4, 5, 80
    num_prior = 3
    
    
    # 3) layer3-cls-confidence
    net['split_layer3_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD3_1')(net['split_layer3'])
    try:
        net['split_layer3_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer3_conf_Dep'])
    except:
        net['split_layer3_conf_Dep'] = tf.nn.relu(net['split_layer3_conf_Dep'])
    net['split_layer3_conf_Conv'] = Conv2D(num_prior * 2 * num_classes, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD3_2')(net['split_layer3_conf_Dep'])
    # net['split_layer3_conf_Reshape'] = Reshape((120, 2))(net['split_layer3_conf_Conv'])
    net['split_layer3_conf_Reshape'] = Reshape((120, num_classes))(net['split_layer3_conf_Conv'])
    #    layer3-bbox, 4是x, y, h, w的调整
    net['split_layer3_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD3_1')(net['split_layer3'])
    try:
        net['split_layer3_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer3_loc_Dep'])
    except:
        net['split_layer3_loc_Dep'] = tf.nn.relu(net['split_layer3_loc_Dep'])
    net['split_layer3_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD3_2')(net['split_layer3_loc_Dep'])
    # net['split_layer3_loc_Reshape'] = Reshape((120, 1, 4))(net['split_layer3_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer3_loc_Reshape'] = Reshape((120, 4))(net['split_layer3_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer4']的通道进行处理
    # 2, 3, 64
    num_prior = 3
    
    
    # 4) layer4-cls-confidence
    net['split_layer4_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD4_1')(net['split_layer4'])
    try:
        net['split_layer4_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer4_conf_Dep'])
    except:
        net['split_layer4_conf_Dep'] = tf.nn.relu(net['split_layer4_conf_Dep'])
    net['split_layer4_conf_Conv'] = Conv2D(num_prior * 2* num_classes, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD4_2')(net['split_layer4_conf_Dep'])
    # net['split_layer4_conf_Reshape'] = Reshape((36, 2))(net['split_layer4_conf_Conv'])
    net['split_layer4_conf_Reshape'] = Reshape((36, num_classes))(net['split_layer4_conf_Conv'])
    #    layer4-bbox, 4是x, y, h, w的调整
    net['split_layer4_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD4_1')(net['split_layer4'])
    try:
        net['split_layer4_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer4_loc_Dep'])
    except:
        net['split_layer4_loc_Dep'] = tf.nn.relu(net['split_layer4_loc_Dep'])
    net['split_layer4_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_loc_DD4_2')(net['split_layer4_loc_Dep'])
    # net['split_layer4_loc_Reshape'] = Reshape((36, 1, 4))(net['split_layer4_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer4_loc_Reshape'] = Reshape((36, 4))(net['split_layer4_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 对net['split_layer5']的通道进行处理
    # 1, 1, 64
    num_prior = 3
    
    
    # 5) layer5-cls-confidence
    net['split_layer5_conf_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_conf_DD5_1')(net['split_layer5'])
    try:
        net['split_layer5_conf_Dep'] = KL.ReLU(max_value=6.)(net['split_layer5_conf_Dep'])
    except:
        net['split_layer5_conf_Dep'] = tf.nn.relu(net['split_layer5_conf_Dep'])
    net['split_layer5_conf_Conv'] = Conv2D(num_prior * 2 * num_classes, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_conf_DD5_2')(net['split_layer5_conf_Dep'])
    # net['split_layer5_conf_Reshape'] = Reshape((6, 2))(net['split_layer5_conf_Conv'])
    net['split_layer5_conf_Reshape'] = Reshape((6, num_classes))(net['split_layer5_conf_Conv'])
    
    #    layer5-bbox, 4是x, y, h, w的调整
    net['split_layer5_loc_Dep'] = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, name='DepthwiseConv2D_loc_DD5_1')(net['split_layer5'])
    try:
        net['split_layer5_loc_Dep'] = KL.ReLU(max_value=6.)(net['split_layer5_loc_Dep'])
    except:
        net['split_layer5_loc_Dep'] = tf.nn.relu(net['split_layer5_loc_Dep'])
    net['split_layer5_loc_Conv'] = Conv2D(2 * num_prior * 4, (1, 1), padding='same', use_bias=True, strides=(1, 1),name='Conv2D_loc_DD5_2')(net['split_layer5_loc_Dep'])
    # net['split_layer5_loc_Reshape'] = Reshape((6, 1, 4))(net['split_layer5_loc_Conv'])  # num_classes-1, 去除背景类别；
    net['split_layer5_loc_Reshape'] = Reshape((6, 4))(net['split_layer5_loc_Conv'])  # num_classes-1, 去除背景类别；

    # 将所有结果进行堆叠
    net['cls_conf'] = Concatenate(axis=1, name='mbox_conf')([net['split_layer1_conf_Reshape'],
                                                              net['split_layer2_conf_Reshape'],
                                                              net['split_layer3_conf_Reshape'],
                                                              net['split_layer4_conf_Reshape'],
                                                              net['split_layer5_conf_Reshape']])
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['split_layer1_loc_Reshape'],
                                                            net['split_layer2_loc_Reshape'],
                                                            net['split_layer3_loc_Reshape'],
                                                            net['split_layer4_loc_Reshape'],
                                                            net['split_layer5_loc_Reshape']])

    # 1242,2
    net['cls_conf'] = Activation('softmax', name='cls_conf_final')(net['cls_conf'])
    # 1242,4
    net['mbox_loc'] = Reshape((1242, 4), name='mbox_loc_final')(net['mbox_loc'])

#分别训练
    # model_return
    # net_output = []
    # net_output.append(net['mbox_loc'])
    # net_output.append(net['cls_conf'])

    # model = Model(input_tensor, net_output)
    
# 一起训练
    # 1242,6
    net['predictions']  = Concatenate(axis =-1, name='predictions')([net['mbox_loc'], net['cls_conf']])

    
    model = Model(input_tensor, net['predictions'])
    # model = Model(input_tensor, [net['mbox_loc'], net['cls_conf']])
    
    return model