import tensorflow as tf

def analyze_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    details = interpreter.get_tensor_details()
    int_ops = 0
    float_ops = 0
    
    for op_details in details:
        if op_details['dtype'] == tf.int8 or op_details['dtype'] == tf.int32:
            int_ops += 1
        elif op_details['dtype'] == tf.float32:
            float_ops += 1
    
    print(f"整型、浮点操作数：{int_ops, float_ops}")
    

model_path = r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/class_08_test_model_2_less.tflite"
analyze_tflite_model(model_path=model_path)