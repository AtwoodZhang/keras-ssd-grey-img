import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Concatenate, ReLU6, Reshape, Identity


def channel_shuffle(x, groups):
    height, width, channels = x.shape[1:]
    channels_per_group = channels // groups
    # Reshape
    x = Reshape([height, width, groups, channels_per_group])(x)
    # Swap the channels and group axes
    x = Reshape([height, width, channels])(x)  # Concatenate the groups back together
    return x


def shufflenet_unit(x, out_channels, groups=2, stride=1):
    in_channels = x.shape[-1]
    if stride == 2:
        # If stride == 2, we create two branches, one for downsampling
        x1 = DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
        x1 = Conv2D(out_channels // 2, (1, 1), padding='same', use_bias=False)(x1)
        x1 = ReLU6()(x1)
        x2 = Conv2D(out_channels // 2, (1, 1), padding='same', use_bias=False)(x)
        x2 = ReLU6()(x2)
        x2 = DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x2)
        x2 = Conv2D(out_channels // 2, (1, 1), padding='same', use_bias=False)(x2)
        x2 = ReLU6()(x2)
        out = Concatenate()([x1, x2])
    else:
        # If stride == 1, we use a residual connection
        residual = x
        x = Conv2D(out_channels, (1, 1), padding='same', use_bias=False)(x)
        x = ReLU6()(x)
        x = DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
        x = Conv2D(out_channels, (1, 1), padding='same', use_bias=False)(x)
        if in_channels == out_channels:
            out = Add()([residual, x])
        else:
            out = x
    out = channel_shuffle(out, groups)
    out = ReLU6()(out)
    return out


def model_version_sh2():
    input_layer = tf.keras.layers.Input(shape=(120, 160, 1))
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu')(input_layer)
    x = shufflenet_unit(x, output_channels=16, stride=1)
    x = shufflenet_unit(x, output_channels=32)
    x = shufflenet_unit(x, output_channels=32, stride=1)
    x = shufflenet_unit(x, output_channels=32)
    x = shufflenet_unit(x, output_channels=16)
    x = shufflenet_unit(x, output_channels=16)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    # model.summary()
    return model