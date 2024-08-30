import time
import cv2
import os
import numpy as np
from PIL import Image
from ssd_pred import SSD
from tqdm import tqdm
from utils import voc_evaluate

if __name__ == "__main__":
    # ssd = SSD(
    #     model_path ='output/20240228/hand_detection_20240626_01.h5',
    #     classes_path ='model_data/voc_classes.txt',
    # )
    run_tflite = True
    model_path = "models/pc_screen_681_qat_norm_de_aug_2rd.tflite"
    use_qat = True
    decouple_heads = True

    output_dir = "results/val"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ssd = SSD()
    mode = "predict" # or dir_predict  # dir predict用来预测文件夹，predict用来预测文件图片
    crop = False  # 指定了是否在单张图片预测后对目标进行截取
    count = False  # 指定了是否进行目标的计数
    # dir_origin_path= "img/"  # 指定了用于检测的图片的文件夹路径
    # dir_save_path = "img_out/"  # 指定了检测完图片的保存路径
    txt_dir = "datasets/VOCdevkit/ImageSets/Main/val.txt"
    img_dir = "datasets/VOCdevkit/JPEGImages"
    with open(txt_dir, "r") as f:
        img_list = f.readlines()
    img_list = [i.strip() for i in img_list]

    with open("val.txt", "r") as fb:
        gt_labels = fb.readlines()
    gt = {}
    for gt_label in gt_labels:
        line = gt_label.split()
        name = line[0].split("/")[-1]
        gt[name] = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    ssd.gt_labels = gt

    # pred_img_path = "datasets/VOCdevkit/JPEGImages/240809_151300_00214.png"
    for img_path in tqdm(img_list):
        pred_img_path = f"{img_dir}/{img_path}.png"
        if mode == "predict":
            img = pred_img_path
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
            else:
                r_image = ssd.detect_image(image, run_tflite, use_qat, model_path, decouple_heads, crop=crop, count=count, image_file=f"{img_path}.png")
                r_image.save(f"{output_dir}/{img_path}.jpg")
    recall, precision, ap = voc_evaluate(ssd.pred_labels, ssd.gt_labels)
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"AP50: {ap}")