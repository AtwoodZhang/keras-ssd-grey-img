import keras.backend as K
from keras.layers import Conv2D, DepthwiseConv2D, add
import keras
import tensorflow as tf


# block of DepthwiseConv2D and Conv2D
def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1,
                          activation="relu"):
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2, strides=strides, use_bias=True,
                        activation=activation, name='block_%d_conv_dw' % block_id)(inputs)
    x = Conv2D(pointwise_conv_filters, kernel_size=(1, 1), padding="same", use_bias=True, strides=strides,
               name='block_%d_conv_pw' % block_id)(x)
    x = add([inputs, x])
    return x


def mobilenet(inputs_tensor):
    # --------------------------主干特征提取网络开始--------------------------#
    # SSD结构，net字典
    net = {}
    # inputs_tensor: 120 * 160 * 1 --> 120 * 160 * 8
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=8, strides=(1, 1), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer1')(inputs_tensor)
    # --> 60 * 80 * 8 
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(2, 2), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer2')(x)
    # --> 60 * 80 * 8
    x = Conv2D(8, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation='relu', name='Conv2D_layer3')(x)
    # x = Conv2D(8, (1, 1), padding='same', use_bias=True, strides=(1, 1), name='Conv2D_layer3')(x)
    # --> 30 * 40 * 16                               
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2, strides=(2, 2), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer4')(x)
    # --> 30 * 40 * 8  
    x = Conv2D(8, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer5')(x)

    # Block 1~3  
    x = _depthwise_conv_block(x, 8, 1, block_id=1)  # --> 30 * 40 * 8
    x = _depthwise_conv_block(x, 8, 1, block_id=2)  # --> 30 * 40 * 8
    x = _depthwise_conv_block(x, 8, 1, block_id=3)  # --> 30 * 40 * 8             

    # Conv_Depth_Conv_Depth_Conv
    # --> 30 * 40 * 16
    x = Conv2D(16, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer6')(x)
    # --> 30 * 40 * 16
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer7')(x)
    # --> 30 * 40 * 16
    x = Conv2D(16, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer8')(x)
    # --> 15 * 20 * 32
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2, strides=(2, 2), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer9')(x)
    # --> 15 * 20 * 24 
    x = Conv2D(24, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer10')(x)

    # Block 4~6
    x = _depthwise_conv_block(x, 24, 1, block_id=4)  # --> 15 * 20 * 24
    x = _depthwise_conv_block(x, 24, 1, block_id=5)  # --> 15 * 20 * 24
    x = _depthwise_conv_block(x, 24, 1, block_id=6)  # --> 15 * 20 * 24

    # --> 15 * 20 * 48
    x = Conv2D(48, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer11')(x)
    # --> 15 * 20 * 48
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer12')(x)
    # --> 15 * 20 * 64
    x = Conv2D(64, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer13')(x)

    net['split_layer1'] = x

    # start split;
    # --> 8 * 10 * 64
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer14')(x)
    # --> 8 * 10 * 40
    x = Conv2D(40, (1, 1), padding='same', use_bias=True, strides=(2, 2), activation="relu", name='Conv2D_layer15')(x)

    # block 7_1 ~7_2
    x = _depthwise_conv_block(x, 40, 1, block_id=7)  # --> 8 * 10 * 40
    x = _depthwise_conv_block(x, 40, 1, block_id=8)  # --> 8 * 10 * 40

    # --> 8 * 10 * 80
    x = Conv2D(80, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer16')(x)
    # --> 8 * 10 * 80
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(1, 1), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer17')(x)
    # --> 8 * 10 * 80
    x = Conv2D(80, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer18')(x)

    net['split_layer2'] = x

    # --> 4 * 5 * 80
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(2, 2), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer19')(x)
    # --> 4 * 5 * 80
    x = Conv2D(80, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer20')(x)

    net['split_layer3'] = x

    # --> 2 * 3 * 80
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(2, 2), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer21')(x)

    # --> 2 * 3 * 64
    x = Conv2D(64, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer22')(x)

    net['split_layer4'] = x

    # --> 1 * 1 * 64
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, strides=(3, 3), use_bias=True, activation="relu",
                        name='DepthWiseConv2D_layer23')(x)
    x = Conv2D(64, (1, 1), padding='same', use_bias=True, strides=(1, 1), activation="relu", name='Conv2D_layer24')(x)


    net['split_layer5'] = x

    # -------------------------------------主干特征提取网络结束--------------------------------#
    return net
