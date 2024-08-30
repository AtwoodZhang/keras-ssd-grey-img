import os 
import tensorflow as tf
import numpy as np
print("Tensorflow version:", tf.__version__)
from PIL import Image

# 制作量化数据集
path = r"/home/zhangyouan/桌面/zya/dataset/681/srp/RockSecissorsPaper_enlarge/train/"
list_dir = os.listdir(path)

labels = {"paper":0, "rock":1, "scissors":2}

test_images = []
test_images_link = []
test_labels = []
for i in list_dir:
    path1 = path + i + "/"
    list_label = os.listdir(path1)
    for j in list_label:
        path2 = path1 + j
        tmp = [0, 0, 0]
        tmp[labels[i]]=1
        test_labels.append(tmp)
        test_images_link.append(path2)
        test_images_tmp = Image.open(path2)
        test_images_g = test_images_tmp.convert('L')
        test_images_g_resize = test_images_g.resize((160, 120), Image.ANTIALIAS) # (width, height)
        test_images.append(np.array(test_images_g_resize))

test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_images = np.expand_dims(test_images, axis=-1)
test_images = test_images.astype(np.float32) # / 255.0
test_images = test_images / 255.0  # 将数值范围压缩到0~1之间；
# test_images = test_images/127.5-1

model = tf.keras.models.load_model(r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/class_08_test_model_2_less.h5")
# model.summary()
scores = model.evaluate(test_images, test_labels, verbose=0)
print("模型准确度：",scores[1])