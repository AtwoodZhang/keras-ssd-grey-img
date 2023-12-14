import numpy as np
import os
import copy

def anchor_from_yxhw2xyxy_less_1(anchor_file):
    anchor_yxhw = anchor_file
    anchor_xyxy = copy.deepcopy(anchor_yxhw)
    print(anchor_yxhw[0][0])
    for i, i_value in enumerate(anchor_yxhw):   
        anchor_xyxy[i][0] = anchor_yxhw[i][1] - 0.5 * anchor_yxhw[i][3]
        anchor_xyxy[i][1] = anchor_yxhw[i][0] - 0.5 * anchor_yxhw[i][2]
        anchor_xyxy[i][2] = anchor_yxhw[i][1] + 0.5 * anchor_yxhw[i][3]
        anchor_xyxy[i][3] = anchor_yxhw[i][0] + 0.5 * anchor_yxhw[i][2]
        anchor_xyxy[i][0] /= 160
        anchor_xyxy[i][2] /= 160
        anchor_xyxy[i][1] /= 120
        anchor_xyxy[i][3] /= 120
    anchor_boxes = np.minimum(np.maximum(anchor_xyxy, 0.0), 1.0)
    return anchor_boxes


if __name__ == "__main__":
    anchor_path = r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback/sony_detection_anchor_boxes.txt"
    anchor = []
    with open(anchor_path, 'r') as f:
        for line in f:
            anchor.append(line.strip().split(', '))
        print(anchor)
    for i, i_value in enumerate(anchor):
        for j, j_value in enumerate(i_value):
            try:
                anchor[i][j] = np.float(anchor[i][j])
            except:
                anchor[i][j] = np.float(anchor[i][j][0:-1])
    
    yxhw_anchor = anchor
    xyxy_anchor = anchor_from_yxhw2xyxy_less_1(yxhw_anchor)  # original anchor is yxhw
    print(xyxy_anchor)
    
    