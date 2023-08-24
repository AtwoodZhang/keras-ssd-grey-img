import numpy as np

from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


# image_size = (120,120)
image_size = (160, 120)
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2


# #TODO：681
spec = [
    SSDSpec((15,20), (8,8), SSDBoxSizes(32, 59), [1]),
    SSDSpec((8,10), (15,16), SSDBoxSizes(59, 86), [2, 3]),
    SSDSpec((4,5), (30,32), SSDBoxSizes(86, 113), [2, 3]),
    SSDSpec((2,3), (60,53),SSDBoxSizes(113, 141), [2, 3]),
    SSDSpec((1,1), (120,160), SSDBoxSizes(141, 168), [2, 3])
]

priors = generate_ssd_priors(spec, image_size)