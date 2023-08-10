import copy
from Anchors import get_anchors
from pprint import pprint


def round_list(input_list, decimals=0):
    return [[int(round(value, decimals)) for value in row] for row in input_list]


input_shape = [120, 160]
anchors_size=[32, 59, 86, 113, 141, 168]
img_width = input_shape[1]
img_height = input_shape[0]
anchors = get_anchors(input_shape, anchors_size)
anchors[:, ::2] *= img_width
anchors[:, 1::2] *= img_height
xywh_anchors = copy.deepcopy(anchors)

# 将x, y交换，按照681要求，改为：y, x, w, h
xywh_anchors[:, 3::4] = anchors[:, 3::4] - anchors[:, 1::4]  # box_height
xywh_anchors[:, 2::4] = anchors[:, 2::4] - anchors[:, ::4]  # box_width
xywh_anchors[:, 0::4] = anchors[:, 1::4] + (0.5*xywh_anchors[:, 3::4])
xywh_anchors[:, 1::4] = anchors[:, ::4] + (0.5*xywh_anchors[:, 2::4])

print(xywh_anchors)

rounded_anchors = round_list(xywh_anchors, decimals=0)
print("四舍五入之后：")
pprint(rounded_anchors)


anchors_txt_path = "./anchors.txt"
with open(anchors_txt_path, 'w') as f:
    for r in rounded_anchors:
        row_str = ','.join(map(str, r))
        f.write(row_str + '\n')
print("write to: " + anchors_txt_path + "  finish")

