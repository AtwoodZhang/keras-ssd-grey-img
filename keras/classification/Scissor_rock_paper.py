import sys
sys.path.append('./keras/classification/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息；
import zipfile
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from Models import Sample_model
from Models import model_version1


def cls_model_build(data_path=r"/home/zhangyouan/桌面/zya/dataset/681/srp/RockSecissorsPaper_enlarge/", 
                    epoch=100, 
                    save_model_name = 'class_08_test_model_2_less.h5',
                    model_summary="True",
                    weights = None,
                    model_load = 2):
    if model_load == 1:
        model = model_version1()
    elif model_load == 2:
        model = Sample_model(model_summary=model_summary)
        
    if weights is not None:
        model.load_weights(weights)
        
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
    
    #======== 模型参数编译 =========
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer = sgd,
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )
    # model.compile(
    #     optimizer = 'rmsprop',
    #     loss = 'categorical_crossentropy', # 损失函数： 稀疏的交叉熵 binary_crossentropy
    #     metrics = ['accuracy']
    # )
    
    #======== 模型训练 =========
    # Note that this may take some time.
    history = model.fit(
        training_generator,
        epochs = epoch,
        validation_data = validation_generator,
    )

    model.save(save_model_name) # model 保存

    return history
    
    

if __name__ == "__main__":
    from log_visualization_tool import visual_train
    history = cls_model_build(data_path=r"/home/zhangyouan/桌面/zya/dataset/681/srp/RockSecissorsPaper_enlarge/", 
                            epoch=100, 
                            save_model_name = './../../output/keras/cls/model_v0_m1.h5' )
    visual_train(history)