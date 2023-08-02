import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息；
import zipfile
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator



def cls_model_v0(data_path=r"/home/zhangyouan/桌面/zya/dataset/681/srp/RockSecissorsPaper_enlarge/", 
                 epoch=100, 
                 save_model_name = 'class_08_test_model_2_less.h5' ):
    
    train_data_path = data_path + "train"
    test_data_path = data_path + "val"
        
    """step1. create dataset generator.
    """
    training_datagen = ImageDataGenerator(
        # 数据增强
        rescale=1. / 255,
        rotation_range=40, # 旋转范围
        width_shift_range=0.2, # 宽平移
        height_shift_range=0.2,# 高平移
        shear_range=0.2, # 剪切
        zoom_range=0.2, # 缩放
        horizontal_flip=True,
        fill_mode='nearest'    
    )

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    TRAINING_DIR = train_data_path
    VALIDATION_DIR = test_data_path

    training_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(120, 160),
        color_mode="grayscale",
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(120, 160),
        color_mode="grayscale",
        class_mode='categorical'
    )


    """step2. build the model
    """
    #======== 模型构建 =========
    model = tf.keras.models.Sequential([
        # model 1
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (120, 160, 1)), # 输入参数：过滤器数量，过滤器尺寸，激活函数：relu， 输入图像尺寸
        tf.keras.layers.MaxPooling2D(2, 2), # 池化：增强特征
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'), # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'), # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'), # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'), # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
        # tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'), # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
        # tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
        # tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(), # 输入层
        # tf.keras.layers.Dense(1, activation = 'relu'), # 全连接隐层 神经元数量：128 ，激活函数：relu
        tf.keras.layers.Dense(3, activation = 'softmax') # 英文字母分类 26 ，阿拉伯数字分类 10  输出用的是softmax 概率化函数 使得所有输出加起来为1 0-1之间
    ])

    model.summary()

    #======== 模型参数编译 =========
    model.compile(
        optimizer = 'rmsprop',
        loss = 'categorical_crossentropy', # 损失函数： 稀疏的交叉熵 binary_crossentropy
        metrics = ['accuracy']
    )
    
    #======== 模型训练 =========
    # Note that this may take some time.
    history = model.fit(
        training_generator,
        epochs = epoch,
        validation_data = validation_generator,
    )

    model.save(save_model_name) # model 保存

    return history