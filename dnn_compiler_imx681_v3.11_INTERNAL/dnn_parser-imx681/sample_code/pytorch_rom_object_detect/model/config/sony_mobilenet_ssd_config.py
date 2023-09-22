import numpy as np

from utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = (120, 160)
image_mean = np.array([127])  
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec((15, 20), 8, SSDBoxSizes(12, 30), [2]),
    SSDSpec((8, 10), 15, SSDBoxSizes(30, 48), [2,3]),
    SSDSpec((4, 5), 30, SSDBoxSizes(48, 66), [2,3]),
    SSDSpec((2, 3), 60, SSDBoxSizes(66, 84), [2,3]),
    SSDSpec((1, 1), 120, SSDBoxSizes(84, 102), [2,3])
]

priors = generate_ssd_priors(specs, image_size)
