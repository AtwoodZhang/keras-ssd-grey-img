import sys
sys.path.append('./')

import datetime
import os
import keras.backend as K
import tensorflow as tf
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import Conv2D, Dense, DepthwiseConv2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.utils.multi_gpu_utils import multi_gpu_model
from nets.ssd import SSD300
from nets.ssd_training import MultiboxLoss, get_lr_scheduler
from utils.anchors import get_anchors
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ParallelModelCheckpoint, EvalCallback)
from utils.dataloader import SSDDatasets
from utils.utils import get_classes, show_config

tf.logging.set_verbosity(tf.logging.ERROR)

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
    # ---------------------------------------------------------------------#
    #   train_gpu   训练用到的GPU
    #               默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
    #               在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
    # ---------------------------------------------------------------------#
    train_gpu = [0, ]
    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    classes_path = '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_GOOD_Gest/model_data/voc_classes.txt'
    model_path = '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_GOOD_Gest/trained_model/20230808_orig_test.h5'
    # ------------------------------------------------------#
    #   input_shape     输入的shape大小
    # ------------------------------------------------------#
    input_shape = [120, 160]
    # ------------------------------------------------------#
    # anchors_size    = [30, 60, 111, 162, 213, 264, 315]
    anchors_size = [32, 59, 86, 113, 140, 160]
    # anchors_size = [15, 40, 80, 113, 140, 168]
    # anchors_size = [21, 32, 59, 86, 113, 153]

    Init_Epoch = 0
    batch_size = 16
    Epochs = 500

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=6e-4
    #                   当使用SGD优化器时建议设置   Init_lr=2e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=6e-4
    #                   当使用SGD优化器时建议设置   Init_lr=2e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 2e-3
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 10
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_GOOD_Gest/logs'
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 10
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = 1

    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    # ------------------------------------------------------#
    train_annotation_path = '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/VOC_dataset/2007_train.txt'
    val_annotation_path = '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/VOC_dataset/2007_val.txt'

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, anchors_size)

    K.clear_session()
    model_body = SSD300((input_shape[0], input_shape[1], 1), num_classes)
    if model_path != '':
        # ------------------------------------------------------#
        #   载入预训练权重
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)


    model = model_body

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Epoch=Epochs, batch_size=batch_size, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )
    # ---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    # ----------------------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // batch_size * Epochs
    if total_step <= wanted_step:
        if num_train // batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
        num_train, batch_size, Epochs, total_step))
        print(
            "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (total_step, wanted_step, wanted_epoch))

    # for layer in model_body.layers:
    #     if isinstance(layer, DepthwiseConv2D):
    #         layer.add_loss(l2(weight_decay)(layer.depthwise_kernel))
    #     elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
    #         layer.add_loss(l2(weight_decay)(layer.kernel))

    if True:
        batch_size = batch_size
        start_epoch = Init_Epoch
        end_epoch = Epochs

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-5
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        Min_lr_fit = 1e-3

        optimizer = {
            'adam': Adam(lr=Init_lr_fit, beta_1=momentum),
            # 'sgd': SGD(lr=Init_lr_fit, momentum=momentum, nesterov=True)
            'sgd': SGD(lr=0.001, momentum=momentum, nesterov=True)
        }[optimizer_type]
        model.compile(optimizer=optimizer, loss=MultiboxLoss(num_classes, neg_pos_ratio=3.0).compute_loss)

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epochs)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataloader = SSDDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train=False, overlap_threshold=0.5)
        val_dataloader = SSDDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train=False, overlap_threshold=0.5)

        # -------------------------------------------------------------------------------#
        #   训练参数的设置
        #   logging         用于设置tensorboard的保存地址
        #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
        #   lr_scheduler    用于设置学习率下降的方式
        #   early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
        # -------------------------------------------------------------------------------#
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        logging = TensorBoard(log_dir)
        loss_history = LossHistory(log_dir)
        checkpoint = ModelCheckpoint(
            os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
            monitor='val_loss', save_weights_only=True, save_best_only=False, period=save_period)
        checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"),
                                          monitor='val_loss', save_weights_only=True, save_best_only=False,
                                          period=1)
        checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"),
                                          monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
        eval_callback = EvalCallback(model_body, input_shape, anchors, class_names, num_classes, val_lines, log_dir, \
                                     eval_flag=eval_flag, period=eval_period)
        callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, 
                    #  lr_scheduler, 
                     eval_callback]

        if start_epoch < end_epoch:
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                generator=train_dataloader,
                steps_per_epoch=epoch_step,
                validation_data=val_dataloader,
                validation_steps=epoch_step_val,
                epochs=end_epoch,
                initial_epoch=start_epoch,
                use_multiprocessing=True if num_workers > 1 else False,
                workers=num_workers,
                callbacks=callbacks
            )

        # ---------------------------------------#
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        # ---------------------------------------#
        model.save("/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_GOOD_Gest/trained_model/20230808_orig_test.h5")
        model.save("/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_GOOD_Gest/trained_model/20230808_orig_test.pb")
