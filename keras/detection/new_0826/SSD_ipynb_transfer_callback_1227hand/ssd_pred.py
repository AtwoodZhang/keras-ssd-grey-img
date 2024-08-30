import colorsys
import os
import time
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageDraw, ImageFont

from Models import SSD300, SSD300_bk
# from Models_large import SSD300
from Anchors import get_anchors
from utils import get_classes, resize_image, show_config
from Datasets import cvtColor
from utils_bbox import BBoxUtility
import tensorflow as tf
from scipy.special import softmax 


class SSD(object):
    _defaults = {
        # "model_path"        : '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227hand/output/20240228/hand_detection_20240311.h5',
        # "model_path"          : '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_from_GPT_debug/final_detection_model_det_good.h5',
        "model_path": "datasets/VOCdevkit/20240810_pc_screen.h5",
        # "model_path"  : "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227hand/output/20240228/hand_detection_20240626_01.h5",
        # "model_path"  : "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_from_GPT_debug/final_detection_model_det_good_hand.h5",
        "classes_path"      : 'datasets/VOCdevkit/voc_classes.txt',
        # "classes_path"      : '/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_GOOD_Gest/model_data/voc_classes.txt',
        "input_shape"       : [120, 160],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   用于指定先验框的大小
        #---------------------------------------------------------------------#
        'anchors_size' : [32, 59, 86, 113, 141, 168],
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    
    #---------------------------------------------------#
    #   初始化ssd
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.gt_labels = ""
        self.pred_labels = {}
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors                        = get_anchors(self.input_shape, self.anchors_size)
        self.num_classes                    = self.num_classes + 1
        
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)
        self.generate()
        
        show_config(**self._defaults)
    
    def load_weights(self, filepath, by_name=False, skip_mismatch=False, reshape=False):
        import h5py as h5
        with h5.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            if by_name:
                saving.load_weights_from_hdf5_group_by_name(
                f, self.layers, skip_mismatch=skip_mismatch,reshape=reshape)
            else:
                saving.load_weights_from_hdf5_group(f, self.layers, reshape=reshape)
        
    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.ssd = SSD300([self.input_shape[0], self.input_shape[1], 1], self.num_classes)
        
        
        self.ssd.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))
    
    def dequant(self, tensor, detail):
        quantization_params = detail['quantization']
        scale, zero_point = quantization_params
        tensor = tensor.astype(np.float32)
        return scale * (tensor - zero_point)

    def qat_tflite_predict(self, image_data, decouple_heads, tflite_path="models/pc_screen_681_qat_norm_de_aug_2rd.tflite"):
        interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_preserve_all_tensors=True)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        if decouple_heads:
            cls_out = interpreter.get_tensor(output_details[0]['index'])
            det_out = interpreter.get_tensor(output_details[1]['index'])
            return np.concatenate([det_out, cls_out], axis=-1)
        else:
            return interpreter.get_tensor(output_details[0]['index'])

    def tflite_predict(self, image_data, decouple_heads, tflite_path="models/pc_screen_681_aug.tflite"):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        if decouple_heads:
            det_out = interpreter.get_tensor(output_details[0]['index'])
            det_out = self.dequant(det_out, output_details[0])
            cls_out = interpreter.get_tensor(output_details[1]['index'])
            cls_out = self.dequant(cls_out, output_details[1])
            return np.concatenate([det_out, cls_out], axis=-1)
        else:
            output_data = interpreter.get_tensor(output_details[0]['index'])
            return self.dequant(output_data, output_details[0])

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, run_tflite, use_qat, model_path, decouple_heads, crop = False, count = False, image_file=""):
        # 计算输入图片的高和宽
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
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        # image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        # preds      = self.ssd.predict(image_data)
        
        image_data = np.expand_dims(np.array(image_data, dtype='float32'), 0)
        image_data = np.expand_dims(np.array(image_data, dtype='float32'), -1)
        # image_data = image_data / 127.5 - 1.0
        # preds = self.ssd.predict(image_data)
        if not run_tflite:
            image_data = image_data / 127.5 - 1.0
            preds = self.ssd.predict(image_data)
        elif use_qat:
            image_data = image_data - 128
            preds = self.qat_tflite_predict(image_data, decouple_heads, tflite_path=model_path)
        else:
            image_data = image_data.astype(np.int8)
            preds = self.tflite_predict(image_data, decouple_heads, tflite_path=model_path)
        
        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)
        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        pred_labels = []
        
        if len(results[0])<=0:
            return image
        for result in results[0]:
            pred_labels.append([result[1], result[0], result[3], result[2], result[5]])
        self.pred_labels[image_file] = np.array(pred_labels)

        top_label   = np.array(results[0][:, 4], dtype = 'int32')
        top_conf    = results[0][:, 5]
        top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        # font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        font = ImageFont.load_default()
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{:.2f}'.format(score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                # draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.rectangle([left + i, top + i, right - i, bottom - i])
            # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)])
            draw.text(tuple(text_origin), str(label,'UTF-8'), fill="black", font=font)
            del draw

        return image
    
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds      = self.ssd.predict(image_data)
        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            preds      = self.ssd.predict(image_data)
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                    self.input_shape, self.letterbox_image, confidence=self.confidence)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds      = self.ssd.predict(image_data)
        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)
        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if len(results[0])<=0:
            return 

        top_label   = results[0][:, 4]
        top_conf    = results[0][:, 5]
        top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
