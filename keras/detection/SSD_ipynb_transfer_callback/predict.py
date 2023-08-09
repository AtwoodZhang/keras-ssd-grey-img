import time
import cv2
import numpy as np
from PIL import Image
from ssd_pred import SSD


if __name__ == "__main__":
    ssd = SSD()
    mode = "predict" # or dir_predict  # dir predict用来预测文件夹，predict用来预测文件图片
    crop = False  # 指定了是否在单张图片预测后对目标进行截取
    count = False  # 指定了是否进行目标的计数
    # dir_origin_path= "img/"  # 指定了用于检测的图片的文件夹路径
    # dir_save_path = "img_out/"  # 指定了检测完图片的保存路径
    pred_img_path = "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback/img/230317_110146_00000.jpg"
    
    if mode == "predict":
        img = pred_img_path
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            r_image = ssd.detect_image(image, crop=crop, count=count)
            # r_image.show()
            r_image.save("/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback/img_pred_output/img.jpg")