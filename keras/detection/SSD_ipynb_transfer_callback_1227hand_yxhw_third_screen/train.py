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

if __name__ == "__main__":
    
    # 设置训练参数
    Epoch = 1000  # 训练100 epochs
    lr = 1e-3  # Adam优化器，所以较小的学习率
    optimizer_type = "Adam"
    momentum = 0.937
    batch_size = 32
    imgcolor = 'grey'  # imgcolor选“rgb” or “grey”, 则处理图像变单通道或者三通道
    save_dir = "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227hand_yxhw_third_screen/output/PhoneScreen"
    
    # 设置SSD参数
    cls_name_path = r"/home/zhangyouan/桌面/zya/dataset/681/PhoneScreen/voc_classes.txt"  # 导入目标检测类别；
    input_shape = [120, 160]  # 输入的尺寸大小
    anchor_size = [32, 59, 86, 113, 141, 168]  # 用于设定先验框的大小，根据公式计算而来；如果要检测小物体，修改浅层先验框的大小，越小的话，识别的物体越小；    
    train_annotation_path = r'/home/zhangyouan/桌面/zya/dataset/681/PhoneScreen/2007_trainval.txt'  # 训练图片路径和标签
    val_annotation_path = r'/home/zhangyouan/桌面/zya/dataset/681/PhoneScreen/2007_test.txt'  # 验证图片路径和标签
    
    # 1. 获取classes和anchor
    class_names, num_cls = get_classes(cls_name_path)
    num_cls += 1  # 增加一个背景类别
    print("class_names:", class_names, "num_classes:", num_cls)
    
    # 2. 获取anchors, 输出的是归一化之后的anchors
    anchor = get_anchors(input_shape, anchor_size)
    print("type:",type(anchor), "shape:", np.shape(anchor))

    # 3. 模型编译
    K.clear_session()
    model_path = ""
    model_path = "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227hand_yxhw_third_screen/ssd_weights.h5"
    # model_path = "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227_exchange_yxhw/output/240806/20240807_2_command.h5"
    model = SSD300((input_shape[0], input_shape[1], 1), num_cls)
    # model.save("template.h5")
    # model.summary()
    if model_path != "":
        model.load_weights(model_path, by_name = True, skip_mismatch=True)
       
    # 4. 优化器
    # optimizer = Adam(lr = lr, beta_1=momentum)
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    from tensorflow.keras.optimizers import legacy
    optimizer = legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # 5. 导入数据集
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    train_dataloader = SSDDatasets(train_lines, input_shape, anchor, batch_size, num_cls, train=False, imgcolor=imgcolor)
    val_dataloader = SSDDatasets(val_lines, input_shape, anchor, batch_size, num_cls, train=False, imgcolor=imgcolor)
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 6. 编译模型
    # losses = {'mbox_loc_final':MultiboxLoss(num_cls, neg_pos_ratio=3.0).compute_loc_loss,
            #   'cls_conf_final':MultiboxLoss(num_cls, neg_pos_ratio=3.0).compute_conf_loss}
    # losses2 = [MultiboxLoss(num_cls, neg_pos_ratio=3.0).compute_loc_loss,MultiboxLoss(num_cls, neg_pos_ratio=3.0).compute_conf_loss]
    model.compile(optimizer=optimizer, loss = MultiboxLoss(num_cls, neg_pos_ratio=3.0).compute_loss)
    
    # 7. 设计learning rate;
    total_steps = int(Epoch * num_train / batch_size)
    # 7.1 compute the number of warmup batches.
    warmup_epochs = 10
    warmup_steps = int(warmup_epochs * num_train / batch_size)
    # 7.2 create the learning rate scheduler
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=lr,
                                            total_steps=total_steps,
                                            warmup_learning_rate=4e-06,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=20)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    
    # # 8. 精度评价: pending --> 还没构建；
    eval_flag = True
    eval_period = 10
    eval_callback = EvalCallback(model, input_shape, anchor, class_names, num_cls, val_lines, log_dir, eval_flag=eval_flag, period = eval_period)
    show_config(
        classes_path=cls_name_path, model_path=model_path, input_shape=input_shape, \
        Epoch=Epoch, batch_size=batch_size, \
        lr=lr, optimizer_type=optimizer_type, momentum=momentum, \
        num_train=num_train, num_val=num_val
    )
    
    callbacks_list = [
        # 早停回调，keras.callbacks.EarlyStopping(monitor='val_accuracy'， patience=4)
        warm_up_lr, # 学习率的调整
        # 学习率调整方法2. keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto',min_lr=0.000001),
        # Epoch结束回调LearningRateSchrduler(schrduler, verbose=1),
        keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir, 'unetlogs'), update_freq=1000), #参数分别为日志存储路径和每多少step进行一次记录，此处不应取太小，会拖慢训练过程
        # eval_callback,  # 精度评价；
    ]
    # 8. 开始训练；
    history = model.fit_generator(
        generator=train_dataloader,
        steps_per_epoch=epoch_step,
        validation_data=val_dataloader,
        validation_steps=epoch_step_val,
        epochs=Epoch,
        # callbacks = [warm_up_lr]
        callbacks = callbacks_list   
    )  # 使用tensorboard --logdir="" 调用查看loss
    
    record_log(history, filename = os.path.join(save_dir, "unetlogs/log.txt"))
    model.save(os.path.join(save_dir, "20240910_PhoneScreen_ssd_init.h5"))
    model.save(os.path.join(save_dir, "20240910_PhoneScreen_ssd_init.pb"))