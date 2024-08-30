#-----------------------------------------------------------
# Retrieve a list of list result on training and test data
# set for each training epoch
#-----------------------------------------------------------
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import os
import pickle



#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw = image.size[0]
    ih = image.size[1]
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        # new_image = Image.new('RGB', size, (128,128,128))
        new_image = Image.new('L', size)
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
    

def visual_train(history):
    try:
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        #-----------------------------------------------------------
        # Plot training and validation accuracy per epoch
        #-----------------------------------------------------------
        epochs = range(len(acc)) # Get number of epochs
        plt.plot(epochs, acc, 'r', label = "tra_acc")
        plt.plot(epochs ,val_acc, 'b', label = "val_acc")
        plt.title("training and validation accuracy")
        plt.legend(loc=0)
        plt.grid(ls='--')  # 生成网格
        plt.show()
        # 曲线呈直线是因为epochs/轮次太少
    except Exception as e:
        print("no accuracy, only loss.")
        
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(loss)) # Get number of epochs
    #-----------------------------------------------------------
    # Plot training and validation loss per epoch
    #-----------------------------------------------------------
    plt.plot(epochs, loss, 'r', label = "train_loss")
    plt.plot(epochs ,val_loss, 'b', label = "val_loss")
    plt.title("training and validation loss")
    plt.legend(loc=0)
    plt.grid(ls='--')  # 生成网格
    plt.show()
    # 曲线呈直线是因为epochs/轮次太少
    


# 1. 获取类
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default: False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_evaluate(detections, annotations, ovthresh=0.5, use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.

    :param detections: Bounding box detections dictionary, keyed on class id (cid) and image_file,
                       dict[cid][image_file] = np.array([[x1,y1,x2,y2,score], [...],...])
    :param annotations: Ground truth annotations, keyed on image_file,
                       dict[image_file] = np.array([[x1,y1,x2,y2,score], [...],...])
    :param cid: Class ID (0 is typically reserved for background, but this function does not care about the value)
    :param ovthresh: Intersection over union overlap threshold, above which detection is considered as correct,
                       if it matches to a ground truth bounding box along with its class label (cid)
    :param use_07_metric: Whether to use VOC 2007 metric

    :return: recall, precision, ap (average precision)
    """

    # extract ground truth objects from the annotations for this class
    class_gt_bboxes = {}
    npos = 0  # number of ground truth bboxes having label cid
    # annotations keyed on image file names or paths or anything that is unique for each image
    for image_name in annotations:
        # for each image list of objects: [[x1,y1, x2,y2, cid], [], ...]
        R = [obj[:4] for obj in annotations[image_name]]
        bbox = np.array(R)
        # difficult is not stored: take it as 0/false
        difficult = np.array([0] * len(R)).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_gt_bboxes[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # detections' image file names/paths
    det_image_files = []
    confidences = []
    det_bboxes = []
    # detections should be keyed on class_id (cid)
    # class_dict = detections[cid]
    class_dict = detections
    for image_file in class_dict:
        dets = class_dict[image_file]
        for k in range(dets.shape[0]):
            det_image_files.append(image_file)
            det_bboxes.append(dets[k, 0:4])
            confidences.append(dets[k, -1])
    det_bboxes = np.array(det_bboxes)
    confidences = np.array(confidences)

    # number of detections
    num_dets = len(det_image_files)
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)

    if det_bboxes.shape[0] == 0:
        return 0., 0., 0.

    # sort detections by confidence
    sorted_ind = np.argsort(-confidences)
    det_bboxes = det_bboxes[sorted_ind, :]
    det_image_files = [det_image_files[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(num_dets):
        R = class_gt_bboxes[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ## compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            # IoU
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    stp = sum(tp)
    recall = stp / npos
    precision = stp / (stp + sum(fp))

    # compute average precision
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return recall, precision, ap