import copy
from Anchors import get_anchors
from Anchors_orig_size import get_anchors2
from pprint import pprint
import numpy as np


class AnchorConverter(object):
    def __init__(self, input_shape=[120, 160], anchors_size=[32, 59, 86, 113, 140, 168]):
        self.input_shape = input_shape
        self.img_width = self.input_shape[1]
        self.img_height = self.input_shape[0]
        self.anchors_size = anchors_size
    def prepare_anchors(self):
        anchors = get_anchors(input_shape=self.input_shape, anchors_size=self.anchors_size)
        return anchors
    def prepare_no_normalized_anchors(self):
        anchors = get_anchors2(input_shape=self.input_shape, anchors_size=self.anchors_size)
        print(anchors)
        return anchors
        
    def convert_relative_to_absolute(self):
        # ---------------------------------------------#
        #   将相对坐标转换为绝对坐标
        #   输入：
        #       anchors: (m, 4)
        #       anchors的顺序为：x1, y1, x2, y2
        #   输出：
        #       absolute_anchors: (m, 4)
        #       absolute_anchors的顺序为：center_y, center_x, h, w
        # ---------------------------------------------#
        anchors = self.prepare_anchors()
        absolute_anchors = np.zeros_like(anchors)
        absolute_anchors[:, 0] = np.round(0.5 * (anchors[:, 1] + anchors[:, 3]) * self.img_height) # center_y
        absolute_anchors[:, 1] = np.round(0.5 * (anchors[:, 0] + anchors[:, 2]) * self.img_width)  # center_x
        absolute_anchors[:, 2] = np.round((anchors[:, 3] - anchors[:, 1]) * self.img_height)  # h
        absolute_anchors[:, 3] = np.round((anchors[:, 2] - anchors[:, 0]) * self.img_width)  # w

        print(absolute_anchors)
        return absolute_anchors
    
    def written_anchors(self, written_path="./anchors_1225.txt"):
        absolute_anchors = self.convert_relative_to_absolute()
        anchors_txt_path = written_path
        rounded_anchors = np.floor(absolute_anchors).astype(int)
        with open(anchors_txt_path, 'w') as f:
            for r in rounded_anchors:
                row_str = ', '.join(map(str, r))
                f.write("  " + row_str + ',\n')
        print("write to: " + anchors_txt_path + "  finish")
        
m = AnchorConverter()
absolute_anchors = m.written_anchors()
# anchors2 = m.prepare_no_normalized_anchors()