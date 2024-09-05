import tensorflow as tf
import numpy as np

# 加载量化后的 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/class_08_test_model_2_less.tflite")

# 在任何操作之前，分配张量
interpreter.allocate_tensors()

# 获取输入和输出张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 定义一个随机输入样本来执行推理
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)

# 设置输入张量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 执行推理
interpreter.invoke()

# 计算模型所需的算力
def calculate_int8_mops(interpreter):
    total_ops = 0
    for op in interpreter.get_tensor_details():
        try:
            tensor = interpreter.tensor(op['index'])()
            shape = tensor.shape
            num_elements = np.prod(shape)
            total_ops += num_elements
        except ValueError:
            print(f"Tensor data is null for tensor index: {op['index']}. Skipping.")

    return total_ops

# 确保 `allocate_tensors()` 已经运行
mops = calculate_int8_mops(interpreter) / 1e6
print(f"Model requires approximately {mops} Mops (int8).")
