import colorsys
import os
import time

import numpy as np
from PIL import ImageDraw, ImageFont
from nets.ssd import SSD300

from utils.anchors import get_anchors
from utils.utils import cvtColor, get_classes, resize_image, show_config
from utils.utils_bbox import BBoxUtility


class SSD(object):
    _defaults = {
        "model_path"        : '/home/zya/zya/AI/NNet/detection/class_01_mobilenet_ssd/test1_from_bubbliiing/keras_test20230213.h5',
        "classes_path"      : 'model_data/voc_classes.txt',
        "input_shape"       : [120, 160],
        "confidence"        : 0.5,
        "nms_iou"           : 0.45,
        "anchors_size"      : [32, 59, 86, 113, 140, 168],
        "letterbox_image"   : False,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
