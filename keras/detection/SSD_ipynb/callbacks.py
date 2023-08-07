import keras
import os
from utils_bbox import BBoxUtility


class EvalCallback(keras.callbacks.Callback):
    def __init__(self, model_body, input_shape, anchors, class_names, num_classes, 
                 val_lines, map_out_path=".temp_map_out", max_boxes=100, confidence=0.05,
                 nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.model_body = model_body
        self.input_shape = input_shape
        self.anchors = anchors
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period
        self.log_dir = "./"
        
        # ---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        # ---------------------------------------------------------# 
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)
        
        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        pass
        