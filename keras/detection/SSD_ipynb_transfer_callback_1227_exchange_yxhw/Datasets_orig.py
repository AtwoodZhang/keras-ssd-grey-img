import tensorflow as tf
from keras.layers import Conv2D, Dense, DepthwiseConv2D,add
from keras.optimizers import SGD, Adam
import keras.backend as K
import numpy as np
import math
import keras
from PIL import Image
from random import shuffle
from keras import layers as KL


def cvtColor(image, cvt2color='grey'):
    if cvt2color == 'grey':
        if len(np.shape(image)) == 3:
            image = image.convert('L')
    if cvt2color == 'rgb':
            image = image.convert('rgb')
    return image


# this part need more attention.

class SSDDatasets(keras.utils.Sequence):
    # train_dataloader = SSDDatasets(train_lines, input_shape, anchor, batch_size, num_cls, train=True)
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, overlap_threshold=0.4, imgcolor='grey'):
        self.annotation_lines = annotation_lines  # 读取数据集
        self.length = len(self.annotation_lines)  # 计算一共多少条数据 348条
        self.input_shape = input_shape             # (120, 160)
        self.anchors = anchors   # [0:1242]: [[0.,0., 0.279.., 0.24...],...]; (1242,4)            
        self.num_anchors = len(anchors)  # 1242
        self.batch_size = batch_size # 1
        self.num_classes = num_classes # 2
        self.train = train # true
        self.overlap_threshold = overlap_threshold  # 0.4
        self.imgcolor = imgcolor # 'grey'
    
    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))  # 向上取整
    
    def __getitem__(self ,index):
        image_data = []
        box_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size): # (0,16)
            i = i % self.length # 0~347依次循环
            # 训练时进行数据的随机增强，验证时不进行数据的随机增强；
            image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train) 
            
            if len(box) != 0:
                boxes = np.array(box[:,:4], np.float32)
                # 进行归一化，调整到0~1之间
                boxes[:,[0,2]] = boxes[:,[0,2]]/(np.array(self.input_shape[1],np.float32))
                boxes[:,[1,3]] = boxes[:,[1,3]]/(np.array(self.input_shape[0],np.float32))
                # 对真实框的种类进行one hot处理
                one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]  # [0:2] [array([1.]), array([1.])]
                box = np.concatenate([boxes, one_hot_label], axis=-1)
                
            box = self.assign_boxes(box)
            image_data.append(image)
            box_data.append(box)
        box_data = np.array(box_data)  # x1y1x2y2
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = image_data.astype(np.float32) / 127.5 - 1.0  # image_data 归一化
        return image_data, box_data
    
    def on_epoch_end(self):
        shuffle(self.annotation_lines)
        
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
               
    def get_random_data(self, annotation_line, input_shape, jitter=.3, random=True):  # jitter颜色相关
        line = annotation_line.split() # ['/VOCdevkit/VOC2007/JPEGImages/002117.jpg','79,281,202,451,0', '106,128,250,297,0']
        image = Image.open(line[0])
        image = cvtColor(image, cvt2color=self.imgcolor)
        iw, ih = image.size # [375,500]
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        # [79,281,202,451,0], [106,128,250,297,0]
        
        if not random:  # test
            scale = min(w / iw, h / ih)  # 160/375; 120/500
            nw = int(iw * scale)  # 0.24 * 375 = 90
            nh = int(ih * scale)  # 0.24 * 500 = 120
            dx = (w - nw) // 2  # 160-90=70
            dy = (h - nh) // 2  # 120-120=0
            
            #   将图像多余的部分加上灰条
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('L', (w, h))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.uint8)
            
            #   对真实框进行调整
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            return image_data, box  # box: xyxy
        
        # train:对图像进行缩放并且进行长和宽的扭曲, ----------------这个比例应该是有问题的；----------------
        new_ar = iw / ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)  # 随机扭曲程度：1.03924
        scale = self.rand(.25, 2)  # 1.5320
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)  # 245
            nh = int(nw / new_ar)  # 235
        image = image.resize((nw, nh), Image.BICUBIC)
               
        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('L', (w, h)) # w=160,h=120
        new_image.paste(image, (dx, dy))  # paste(self, im, box=None, mask=None) 将一张图片覆盖到另外一张图片的指定位置去
        # 这里是将image（w=160，h=120）贴到new_image(w=160,h=120)的坐标为(dx, dy)的位置，以图片左上角为坐标原点；
        image = new_image
        
        #  翻转图像
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image_data = np.array(image, np.uint8)
        #  对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box
    
    def iou(self, box):# box=[0.375, 0.25, 0.59, 0.59]
        # ---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        # ---------------------------------------------#
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # ---------------------------------------------#
        #   真实框的面积
        # ---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # ---------------------------------------------#
        #   先验框的面积
        # ---------------------------------------------#
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        # ---------------------------------------------#
        #   计算iou
        # ---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):# box=[0.375, 0.25, 0.59, 0.59]
        # ---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors] (1242,)
        #   encoded_box [self.num_anchors, 5]
        # ---------------------------------------------#
        iou = self.iou(box)  # (1242,)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        # ---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测
        # ---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        # ---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本
        # ---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # ---------------------------------------------#
        #   利用iou进行赋值 
        # ---------------------------------------------#
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # ---------------------------------------------#
        #   找到对应的先验框
        # ---------------------------------------------#
        assigned_anchors = self.anchors[assign_mask]

        # ---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        # ---------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # ---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        # ---------------------------------------------#
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])

        # ------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()
  
    def assign_boxes(self, boxes): # boxes=[[0.375, 0.25, 0.59, 0.59, 0, 1.],[0.33, 0.55, 0.518, 0.899, 0, 1.]]
        # ---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-1    的内容为先验框所对应的种类，默认为背景
        #   -1      的内容为当前先验框是否包含目标
        # ---------------------------------------------------#
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0  # self.num_classes --> [0,0] --> [1,0] --> 背景，默认为背景；
        if len(boxes) == 0:  #表示boxes为0， 没有ground truth, 所有的先验框都是背景
            return assignment

        # 当boxes不为0，表示有groundtruth, 所以对每一个真实框都进行iou计算； boxes=[0.375, 0.25, 0.59, 0.59],[0.33, 0.55, 0.518, 0.899]
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # ---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_anchors, 4 + 1]
        #   4是编码后的结果，1为iou
        # ---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        # ---------------------------------------------------#
        #   [num_anchors]求取每一个先验框重合度最大的真实框
        # ---------------------------------------------------#
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # ---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        # ---------------------------------------------------#
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # ---------------------------------------------------#
        #   编码后的真实框的赋值
        # ---------------------------------------------------#
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # ----------------------------------------------------------#
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        # ----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 0
        # assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 5:]
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        # ----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        # ----------------------------------------------------------#
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment
