import numpy as np
import cv2
import copy
import os

from transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


class CustomImagesDataset:

    def __init__(self, root, transform=None, target_transform=None, is_yolo=False, is_eval=False, class_names=None, is_background_in_labels = False, is_gray=True):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.is_gray = is_gray
        if is_eval:
            self.imagepath = f"{self.root}"
            self.ids = [info.split('.')[0] for info in os.listdir(self.imagepath)]

        else:
            self.imagepath = f"{self.root}images/"
            self.annotationpath = f"{self.root}labels/"
            self.image_ids = [info.split('.')[0] for info in os.listdir(self.imagepath)]
            self.ids = [info.split('.')[0] for info in os.listdir(self.annotationpath) if os.path.getsize(self.annotationpath + str(info)) > 0 and info.split('.')[0] in self.image_ids]
            self.image_extension = str(os.listdir(self.imagepath)[0].split('.')[1]).lower()
        if class_names:
            self.class_names = class_names

        self.class_dict = {class_name: i for i,
                           class_name in enumerate(self.class_names)}
        self.yolo_annotation_format = is_yolo
        self.is_background_in_labels = is_background_in_labels

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):

        annotation_file = str(self.root) + "labels/{}.txt".format(image_id)
        
        img_size = self.get_image_size(image_id)
        boxes = []
        labels = []
        with open(annotation_file, 'r') as f:
            for line in f:
                class_id, x1, y1, x2, y2 = line.split()
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                if self.yolo_annotation_format:
                    bbox = self._get_pascal_voc_format(bbox, img_size)
                boxes.append(bbox)
                if self.is_background_in_labels:
                   labels.append(int(float(class_id)))
                else:
                   labels.append(int(float(class_id))+1)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        return boxes, labels

    def _read_image(self, image_id):
        image_file = self.imagepath + str(image_id) + ".{}".format(self.image_extension)
        if self.is_gray:
           image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        else:
           image = cv2.imread(str(image_file))
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_image_size(self, image_id):
        img = self._read_image(image_id)
        return img.shape

    def _get_pascal_voc_format(self, boxes, img_size):

        if self.is_gray:
           dh, dw = img_size
        else:
           dh, dw, c = img_size
        xc, yc, w, h = boxes
        x1 = int(float((xc-w/2) * dw))
        x2 = int(float((xc+w/2) * dw))
        y1 = int(float((yc-h/2) * dh))
        y2 = int(float((yc+h/2) * dh))
        converted_bboxes = [x1, y1, x2, y2]
        return converted_bboxes

    def _get_yolo_bbox_format(self, boxes, img_size):
        if self.is_gray:
           dh, dw = img_size
        else:
           dh, dw, c = img_size
        dh = 1/dh
        dw = 1/dw
        x1, y1, x2, y2 = boxes
        xc = (x1 + x2)/2 * dw
        yc = (y1 + y2)/2 * dh
        w = abs(x2-x1) * dw
        h = abs(y2-y1) * dh
        converted_bboxes = [xc, yc, w, h]
        return converted_bboxes
