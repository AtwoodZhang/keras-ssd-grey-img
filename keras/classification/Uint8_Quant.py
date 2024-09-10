import os
import tensorflow as tf
import numpy as np
from keras import backend as K
from PIL import Image

K.clear_session()
print("Tensorflow version:", tf.__version__)


# 1. 制作量化数据集
path = r"/home/zhangyouan/桌面/zya/dataset/681/PCScreen_Book_PhoneScreen/train/"
list_dir = os.listdir(path)

labels = {"book":0, "PcScreen":1, "PhoneScreen":2}

test_images = []
test_images_link = []
test_labels = []

for i in list_dir:
    path1 = path + i + "/"
    list_label = os.listdir(path1)
    for j in list_label:
        path2 = path1 + j
        test_labels.append(labels[i])
        test_images_link.append(path2)
        test_images_tmp = Image.open(path2)
        test_images_g = test_images_tmp.convert('L')
        test_images_g_resize = test_images_g.resize((160, 120), Image.ANTIALIAS) # (width, height)
        test_images.append(np.array(test_images_g_resize))


test_images = np.array(test_images)
test_labels = np.array(test_labels)
print("test dataset size: ", np.shape(test_images))
print("test label size: ", np.shape(test_labels))
test_images = np.expand_dims(test_images, axis=-1)
print("teste_images shape:", np.shape(test_images))
print("original image data:", np.max(test_images[0]), np.min(test_images[0])) # 判断unsigned
test_images = test_images.astype(np.float32) / 255.0
print("after norm: ", np.max(test_images[0]), np.min(test_images[0]))  # 判断0~1


# 2. uint8量化
h5_model = tf.keras.models.load_model(r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/trained_model/pc_book_phone_0904.h5")

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(test_images).batch(1).take(90):
        yield[input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 使用默认的optimizations标记来量化所有固定参数（例如权重）
converter.representative_dataset = representative_data_gen # 使用浮点回退量化进行转换，转换器可以通过该函数估算所有可变数据的动态范围

converter.target_spec.supported_ops = [tf.int8]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
tflite_model_quant = converter.convert()
tflite_name = r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/trained_model/pc_book_phone_0904_float32int8.tflite"
with open(tflite_name, 'wb') as f:
    f.write(tflite_model_quant)