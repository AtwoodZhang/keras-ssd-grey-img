from torch.utils.data import Dataset
import io
import os
import pdb
import torch
import cv2

class ImageNetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, batch_size, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config =config
        self.batch_size = batch_size
        self.transform = transform

        data_path = self.config["data"]["datapath"]
        self.all_image_dirs = os.listdir(data_path)

        self.data_labels = dict() # {img_path, class_name}

        for dir in self.all_image_dirs:
            img_dir_path = os.path.join(data_path, dir)
            all_images = os.listdir(img_dir_path)
            for img in all_images:
                img_path = os.path.join(img_dir_path, img)
                # pdb.set_trace()
                if img_path in self.data_labels:
                    self.data_labels[img_path].append([dir])
                else:
                    self.data_labels[img_path] = [[dir]]
        
        self.len_dataset = len(self.data_labels)
        self.data_labels_keys = list(self.data_labels)
        # pdb.set_trace()

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pdb.set_trace()
        img_name = self.data_labels_keys[idx]
        image = cv2.imread(img_name)
        class_name = int(self.data_labels[self.data_labels_keys[idx]][0][0]) - 1
        sample = {'image': image, 'classes': class_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample