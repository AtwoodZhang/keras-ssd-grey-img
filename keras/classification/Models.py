import tensorflow as tf
from keras import optimizers
       

def Sample_model(model_summary="False", mode="Train"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (1, 1), input_shape = (120, 160, 1), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros()),
        tf.keras.layers.BatchNormalization(), 
        # tf.keras.layers.ReLU(), 
        tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros()),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.ReLU(), 
        # ============version2 model add this layer=================
        tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros(),activation="relu"),
        # ==========================================================                            
        
        tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros()),
        tf.keras.layers.BatchNormalization(), 
        # tf.keras.layers.ReLU(),  
        tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros()),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.ReLU(), 
 
        tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(), 
        # tf.keras.layers.ReLU(), 
        
        tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.ReLU(), 
        
        tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(), 
        # tf.keras.layers.ReLU(), 
        
        tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.ReLU(), 
        
        tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(), 
        # tf.keras.layers.ReLU(), 
        
        tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.zeros),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.MaxPooling2D(2, 2), # 池化：增强特征
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(3),
        tf.keras.layers.Activation("softmax")
    ])
    
    # if mode != "Train":
    #     model.add(tf.keras.layers.Activation("softmax"))
    if model_summary == "True":
        model.summary()
    return model


def model_version1(model_summary="False"):
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
    if model_summary == "True":
        model.summary()
    return model
        
            