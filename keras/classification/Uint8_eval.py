import os
import tensorflow as tf
import numpy as np
from keras import backend as K
from PIL import Image
import random
K.clear_session()
print("Tensorflow version:", tf.__version__)


# 1. 制作量化数据集
path = r"/home/zhangyouan/桌面/zya/dataset/681/PCScreen_Book_PhoneScreen/train/"
list_dir = os.listdir(path)

# labels = {"paper":0, "rock":1, "scissors":2}
labels = {'PcScreen': 0, 'PhoneScreen': 1, 'book': 2}

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


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  global test_images

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    # if input_details['dtype'] == np.uint8:
    if input_details['dtype'] == np.int8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point
    # test_image = test_image.astype(np.uint8)

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions


test_image_index = random.randint(1, 100)
tflite_model_quant_file = r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/trained_model/pc_book_phone_0904_allint8.tflite"

def evaluate_model(tflite_file, model_type):
  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)

  accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (model_type, accuracy, len(test_images)))


evaluate_model(tflite_model_quant_file, model_type="Quantized")