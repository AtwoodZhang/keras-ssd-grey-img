import copy
from Anchors import get_anchors
from pprint import pprint
import numpy as np


def round_list(input_list, decimals=0):
    return [[int(round(value, decimals)) for value in row] for row in input_list]


input_shape = [120, 160]
# anchors_size=[24, 59, 86, 113, 141, 168]
anchors_size = [32, 59, 86, 113, 141, 168]
img_width = input_shape[1]
img_height = input_shape[0]
anchors = get_anchors(input_shape, anchors_size)  # (x1, y1, x2, y2)
anchors[:, ::2] *= img_width
anchors[:, 1::2] *= img_height
xywh_anchors = copy.deepcopy(anchors)

# # 将x, y交换，按照681要求，改为：y, x, h, w
# xywh_anchors[:, 2::4] = anchors[:, 3::4] - anchors[:, 1::4]  # box_height = y2-y1
# xywh_anchors[:, 3::4] = anchors[:, 2::4] - anchors[:, ::4]  # box_width = x2 - x1

# # xywh_anchors[:, 0::4] = anchors[:, 1::4] + (0.5*xywh_anchors[:, 2::4])  # center_y = y1 + 0.5*box_height
# # xywh_anchors[:, 1::4] = anchors[:, ::4] + (0.5*xywh_anchors[:, 3::4])  # center_x = x1 + 0.5*box_width
# xywh_anchors[:, 0::4] = 0.5*(anchors[:, 3::4] + anchors[:, 1::4])  # center_y = 0.5*(y2+y1)
# xywh_anchors[:, 1::4] = 0.5*(anchors[:, 2::4] + anchors[:, 0::4])  # center_x = 0.5*(x2+x1)


# 将x, y交换，按照681要求，改为：y, x, h, w
xywh_anchors[:, 2] = anchors[:, 3] - anchors[:, 1]  # box_height = y2-y1
xywh_anchors[:, 3] = anchors[:, 2] - anchors[:, 0]  # box_width = x2 - x1

xywh_anchors[:, 0] = 0.5*(anchors[:, 3] + anchors[:, 1])  # center_y = 0.5*(y2+y1)
xywh_anchors[:, 1] = 0.5*(anchors[:, 2] + anchors[:, 0])  # center_x = 0.5*(x2+x1)



# print(xywh_anchors)

# rounded_anchors = xywh_anchors
# rounded_anchors = round_list(xywh_anchors, decimals=0)
# print("四舍五入之后：")

# yxhw_anchors = xywh_anchors
rounded_anchors = np.floor(xywh_anchors).astype(int)
pprint(rounded_anchors)


anchors_txt_path = "./anchors_240702.txt"
with open(anchors_txt_path, 'w') as f:
    for r in rounded_anchors:
        row_str = ', '.join(map(str, r))
        f.write("  " + row_str + ',\n')
print("write to: " + anchors_txt_path + "  finish")

