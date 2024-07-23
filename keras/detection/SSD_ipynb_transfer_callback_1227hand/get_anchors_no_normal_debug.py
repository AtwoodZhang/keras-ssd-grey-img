import copy
from Anchors import get_anchors
from pprint import pprint
import numpy as np


input_shape = [120, 160]
anchors_size = [32, 59, 86, 113, 141, 168]
img_width = input_shape[1]
img_height = input_shape[0]
anchors = get_anchors(input_shape, anchors_size)  # (x1, y1, x2, y2)
ymin = anchors[:, 1]
xmin = anchors[:, 0]
ymax = anchors[:, 3]
xmax = anchors[:, 2]

anchors_x= (xmin + xmax) / 2.0
anchors_y = (ymin + ymax) / 2.0
anchors_w = xmax - xmin
anchors_h = ymax - ymin

anchors_x = np.round(anchors_x * (1 << 7))
anchors_y = np.round(anchors_y * (1 << 7))
anchors_w = np.round(anchors_w * (1 << 7))
anchors_h = np.round(anchors_h * (1 << 7))

output_file = "./anchors_240708.txt"
with open(output_file, 'w') as f:
    for i in range(0, len(anchors)):
        f.write(' {}, {}, {}, {}, \n'.format(
            int(anchors_y[i]), int(anchors_x[i]), int(anchors_h[i]), int(anchors_w[i])
        ))
        