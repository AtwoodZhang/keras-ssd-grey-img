import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def MV2_struct(inputs, expansion_factor=2, output_channels=16, stride=2):
    # 扩展层
    expanded_channels = inputs.shape[-1] * expansion_factor
    x = tf.keras.layers.Conv2D(expanded_channels, (1, 1), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    # 深度卷积 (3x3 Depthwise Convolution)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    # 线性瓶颈 (1x1卷积，不带激活函数)
    x = tf.keras.layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x) 

    # 判断是否使用跳跃连接
    if stride == 1 and inputs.shape[-1] == output_channels:
        x = tf.keras.layers.Add()([inputs, x])

    return x

def model_version_mb2():
    input_layer = tf.keras.layers.Input(shape=(120, 160, 1))
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu')(input_layer)
    x = MV2_struct(x, output_channels=16, stride=1)
    x = MV2_struct(x, output_channels=32)
    x = MV2_struct(x, output_channels=32, stride=1)
    x = MV2_struct(x, output_channels=32)
    x = MV2_struct(x, output_channels=16)
    x = MV2_struct(x, output_channels=16)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model

model = model_version_mb2()

# 创建一个数据张量
x = tf.random.normal((1, 120, 160, 1))

# 将模型转换为计算图
concrete_func = tf.function(model).get_concrete_function(x)
frozen_func = convert_variables_to_constants_v2(concrete_func)
frozen_graph = frozen_func.graph.as_graph_def()

# 使用 TensorFlow Profiler 计算 FLOPs
with tf.Graph().as_default() as graph:
    tf.import_graph_def(frozen_graph, name='')
    flops = tf.compat.v1.profiler.profile(
        graph,
        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())

print(f"FLOPs: {flops.total_float_ops}")
