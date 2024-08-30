import time
import cv2
import numpy as np
from PIL import Image
from ssd_pred import SSD
import tensorflow as tf
from utils import get_classes, resize_image, show_config
from Datasets import cvtColor

index = 130

def dequant(tensor, detail):
    quantization_params = detail['quantization']
    scale, zero_point = quantization_params
    tensor = tensor.astype(np.float32)
    return scale * (tensor - zero_point)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def tflite_fp32(image_data, tflite_path="models/pc_screen_681_20240820_large_fp32.tflite"):
    interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_preserve_all_tensors=True)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    one_conv = interpreter.get_tensor(index)
    return one_conv


def tflite_predict(image_data, tflite_path="models/pc_screen_681_20240820_large.tflite"):
    interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_preserve_all_tensors=True)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    one_conv = interpreter.get_tensor(index)
    one_conv = dequant(one_conv, interpreter.get_tensor_details()[index])
    return  one_conv


pred_img_path = "datasets/VOCdevkit/JPEGImages/240809_153025_01105.png"
image = Image.open(pred_img_path)
image_shape = np.array([image.size[1], image.size[0]])
# image_shape = np.array(np.shape(image)[0:2])
#---------------------------------------------------------#
#   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
image       = cvtColor(image)
#---------------------------------------------------------#
#   给图像增加灰条，实现不失真的resize
#   也可以直接resize进行识别
#---------------------------------------------------------#
image_data = resize_image(image, (160, 120), False)
#---------------------------------------------------------#
#   添加上batch_size维度，图片预处理，归一化。
#---------------------------------------------------------#
# image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

# preds      = self.ssd.predict(image_data)

image_data = np.expand_dims(np.array(image_data, dtype='float32'), 0)
image_data = np.expand_dims(np.array(image_data, dtype='float32'), -1)

fp32_image = image_data / 127.5 - 1.0
fp32_preds = tflite_fp32(fp32_image)

int8_image = image_data.astype(np.int8)
int8_preds = tflite_predict(int8_image)

print(1)