import numpy as np
import tensorflow as tf
import os
from PIL import Image
from keras.models import load_model
import tensorflow_model_optimization as tfmot
import sys
sys.path.append('./')
import numpy as np
import os
import datetime
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, Dense, DepthwiseConv2D,add
from keras.optimizers import SGD, Adam
import numpy as np
import math
import keras
from PIL import Image
from random import shuffle
from keras import layers as KL
from Anchors import get_anchors
from Datasets import SSDDatasets
from learning_rate import WarmUpCosineDecayScheduler
from loss import MultiboxLoss
from Models import SSD300
from utils import get_classes, show_config
from log_record import record_log, read_log
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from callbacks import (ExponentDecayScheduler, LossHistory,
                       ParallelModelCheckpoint, EvalCallback)


from keras.regularizers import l2

# 模型路径和图片路径
h5_model_path = "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227hand/output/1227/20240708_command.h5"
imgs_path = "/home/zhangyouan/桌面/zya/dataset/681/hand/VOCdevkit/VOC2007/JPEGImages/"

# 加载模型
keras_model = load_model(h5_model_path, custom_objects={"compute_loss": None})

# 应用QAT
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(keras_model)

# 编译QAT模型
# q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(),
#                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                       metrics=['accuracy'])
num_cls = 2
from tensorflow.keras.optimizers import legacy
optimizer = legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
q_aware_model.compile(optimizer=optimizer, loss = MultiboxLoss(num_cls, neg_pos_ratio=3.0).compute_loss)

# 准备数据，将这些数据用于校准模型，以确定量化参数
all_img_link = []
all_img = []
list_dir = os.listdir(imgs_path)
for i in list_dir:
    img_path = os.path.join(imgs_path, i)
    all_img_link.append(img_path)
    all_img_tmp = Image.open(img_path).convert('L')  # 转为灰度
    all_img_tmp = all_img_tmp.resize((160, 120), Image.ANTIALIAS)  # (width, height)
    all_img.append(np.array(all_img_tmp))

input_data = np.array(all_img)
input_data = np.expand_dims(input_data, axis=-1)
# input_data = input_data.astype(np.float32) / 127.5 - 1.0
input_data = input_data.astype(np.uint8)

print("test dataset size: ", np.shape(input_data))

# 假设有对应的标签数据
train_annotation_path = r'/home/zhangyouan/桌面/zya/dataset/681/hand/2007_train.txt'  # 训练图片路径和标签
val_annotation_path = r'/home/zhangyouan/桌面/zya/dataset/681/hand/2007_val.txt'  # 验证图片路径和标签

with open(train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()
with open(val_annotation_path, encoding='utf-8') as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)
batch_size=32
input_shape = [120, 160]
anchor_size = [32, 59, 86, 113, 141, 168]
anchor = get_anchors(input_shape, anchor_size)
train_dataloader = SSDDatasets(train_lines, input_shape, anchor, batch_size, num_cls, train=False, imgcolor=imgcolor)
val_dataloader = SSDDatasets(val_lines, input_shape, anchor, batch_size, num_cls, train=False, imgcolor=imgcolor)

# 训练QAT模型
q_aware_model.fit(generator=train_dataloader, 
                  validation_data=val_dataloader, 
                  epochs=5)

# 校准模型，使用校准数据来估计量化参数


def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(input_data).batch(1).take(100):
        yield [input_value]

# 转换为TFLite模型
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# 保存为TFLite文件
output_file = 'quantized_hand_model_qat_20240708_qat.tflite'
with open(output_file, 'wb') as f:
    f.write(tflite_model)
    print("has been written to:", output_file)
