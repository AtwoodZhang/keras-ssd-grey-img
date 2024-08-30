import tensorflow as tf
from keras import optimizers
       

def custom_block(inputs):
    # 第一分支
    branch1 = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(inputs)
    branch1 = tf.keras.layers.Conv2D(16, (1, 1), padding='same', activation='relu')(branch1)
    branch1 = tf.keras.layers.DepthwiseConv2D((3, 3),strides=2, padding='same')(branch1)
    
    # 第二分支
    branch2 = tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding='same')(inputs)
    
    # 合并分支
    combined = tf.keras.layers.Add()([branch1, branch2])
    combined = tf.keras.layers.ReLU()(combined)
    
    return combined

def custom_block2(inputs):
    # 第一分支
    branch1 = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(inputs)
    branch1 = tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu')(branch1)
    branch1 = tf.keras.layers.DepthwiseConv2D((3, 3),strides=2, padding='same')(branch1)
    branch1 = tf.keras.layers.BatchNormalization()(branch1)
    # branch1 = tf.keras.layers.ReLU()(branch1)
    
    # 第二分支
    branch2 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same')(inputs)
    branch2 = tf.keras.layers.BatchNormalization()(branch2)
    # branch2 = tf.keras.layers.ReLU()(branch2)
    
    # 合并分支
    combined = tf.keras.layers.Add()([branch1, branch2])
    combined = tf.keras.layers.ReLU()(combined)
    
    return combined


def model_version2():
    input_layer = tf.keras.layers.Input(shape=(120, 160, 1))
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu')(input_layer)

    x = custom_block(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=2, padding='same', activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, (1, 1), strides=1, padding='same', activation="relu")(x)
    x = custom_block2(x)
    x = custom_block2(x)
    x = custom_block2(x)
    # x = custom_block2(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    model.summary()
    
    return model