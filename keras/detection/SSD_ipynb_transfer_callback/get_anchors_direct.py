import copy
from Anchors_orig_size import get_anchors
from pprint import pprint
import numpy as np


def round_list(input_list, decimals=0):
    return [[int(round(value, decimals)) for value in row] for row in input_list]


input_shape = [120, 160]
anchors_size=[32, 59, 86, 113, 141, 168]

img_width = input_shape[1]
img_height = input_shape[0]
xywh_anchors = get_anchors(input_shape, anchors_size)  # (x1, y1, x2, y2)
yxhw_anchors = copy.deepcopy(xywh_anchors)
yxhw_anchors[:, 2] = xywh_anchors[:, 3]  
yxhw_anchors[:, 3] = xywh_anchors[:, 2]  

yxhw_anchors[:, 0] = xywh_anchors[:, 1]
yxhw_anchors[:, 1] = xywh_anchors[:, 0]  


rounded_anchors = np.floor(yxhw_anchors).astype(int)
pprint(rounded_anchors)

anchors_txt_path = "./anchors_0925.txt"
with open(anchors_txt_path, 'w') as f:
    for r in rounded_anchors:
        row_str = ', '.join(map(str, r))
        f.write("  " + row_str + ',\n')
print("write to: " + anchors_txt_path + "  finish")


