import numpy as np
import copy


class AnchorBox():
    def __init__(self, input_shape=[120, 160], min_size=None, max_size=None, aspect_ratios=None, flip=True):
        self.input_shape = input_shape
        self.min_size = min_size  # minmum anchor_size
        self.max_size = max_size  # maxmum anchor_size
        self.aspect_ratios = []
        for i in aspect_ratios: # 当aspect_ratios = [1]
            self.aspect_ratios.append(i) # self.aspect_ratios = [1]
            self.aspect_ratios.append(1.0/i)  # self.aspect_ratios = [1, 1]
    
    def call(self, layer_shape, mask=None):
        layer_height = layer_shape[0]  # 输入进来的特征层的高
        layer_width = layer_shape[1]  # 输入进来的特征层的宽
        img_height = self.input_shape[0]  # 输入进来的图片的高
        img_width = self.input_shape[1]  # 输入进来的图片的宽
        
        box_widths = []
        box_heights = []
        for i in self.aspect_ratios:  # for i in [1, 1]:
            # 1. 首先添加一个较小的正方形
            if i == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            # 2. 然后添加一个较大的正方形
            elif i == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 3. 接着添加长方形
            elif i != 1:
                box_widths.append(self.min_size * np.sqrt(i))
                box_heights.append(self.min_size / np.sqrt(i))
        
        # 划分特征层，计算所有的anchors
        
        # 获得所有先验框的宽高1/2
        box_widths  = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        
        # 每一个特征层对应的步长
        step_x = img_width / layer_width  # width方向的步长，160/20 = 8 
        step_y = img_height / layer_height  # height方向的步长：120/15 = 8
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)  # np.linspace(0.5*8, 160-0.5*8,20) 定义均匀间隔创建数值序列
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)  # np.linspace(0.5*8, 120-0.5*8,15) 定义均匀间隔创建数值序列
        
        # 构建网格
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        
        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_)) # Numpy的 tile() 函数，就是将原矩阵横向、纵向地复制。
        
        # 计算先验框的宽高
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        anchor_boxes[:, ::4] -= box_widths  # x1
        anchor_boxes[:, 1::4] -= box_heights  # y1
        anchor_boxes[:, 2::4] += box_widths  # x2
        anchor_boxes[:, 3::4] += box_heights  # y2
        
        anchor_yxhw = copy.deepcopy(anchor_boxes)
        # # --------------------------------- #
        # #   将先验框变成小数的形式
        # #   归一化
        # # --------------------------------- #
        # anchor_boxes[:, ::2] /= img_width
        # anchor_boxes[:, 1::2] /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)

        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
        return anchor_boxes


def get_img_output_length(height, width):
    feature_heights = [15, 8, 4, 2, 1]
    feature_widths = [20, 10, 5, 3, 1]
    return np.array(feature_heights), np.array(feature_widths)


# 2. 获取anchors
def get_anchors(input_shape=[120, 160], anchors_size=[32, 59, 86, 113, 140, 168]):
    # (feature_heights = [15, 8, 4, 2, 1], feature_widths = [20, 10, 5, 3, 1])
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    aspect_ratios = [[1], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]  # anchor的长宽比
    anchors = []
    for i in range(len(feature_heights)): # range(5)
        anchors.append(AnchorBox(input_shape, anchors_size[i], max_size=anchors_size[i+1], aspect_ratios=aspect_ratios[i])
                       .call([feature_heights[i], feature_widths[i]]))
    anchors = np.concatenate(anchors, axis=0)
    return anchors