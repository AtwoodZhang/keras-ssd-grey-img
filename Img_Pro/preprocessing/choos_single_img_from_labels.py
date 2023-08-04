#----------------------------------------------------------#
# 只提取一个标签，并删除其他标签，并删除不包含该标签的文件；
#----------------------------------------------------------#

import os
import xml.etree.ElementTree as ET

Annotation_path = r"/home/zya/zya/AI/NNet/detection/class_01_mobilenet_ssd/test1_from_bubbliing_github_clone/VOCdevkit/VOC2007/Annotations/"
Image_path = r"/home/zya/zya/AI/NNet/detection/class_01_mobilenet_ssd/test1_from_bubbliing_github_clone/VOCdevkit/VOC2007/JPEGImages/"

def choose_imgandlabel_from_label(label = 'dog'):
    Image_name_list = os.listdir(Image_path)
    Anno_name_list = os.listdir(Annotation_path)
    print(Image_name_list[0:3])
    print(Anno_name_list[0:3])
    
    return


if __name__ == "__main__":
    label1 = "dog"
    choose_imgandlabel_from_label(label = label1)