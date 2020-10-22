import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import os
import glob
import torch
from torch.utils.data import Dataset


class BarCodeDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 is_test=False, keep_difficult=False, label_file=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            self.image_sets_dir = self.root / 'val'
        else:
            self.image_sets_dir = self.root / 'train'
        
        self.ids = BarCodeDataset._read_image_ids(self.image_sets_dir)
        self.keep_difficult = keep_difficult

        self.class_names = ('background', 'bar_code', 'qr_code')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        """
        Return:
            image  -> (B, 3, 300, 300) || tensor(151.) tensor(-117.7494)
            boxes  -> (B, 8732, 4)     || tensor
            labels -> (B, 8732)        || tensor
        """
        image_id = self.ids[index]
        # np.ndarray [num_target, 4] [num_target]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        assert boxes.shape[0] == labels.shape[0], \
            'the num of boxes and labels not equal'
        # label absence
        if len(boxes) == 0 or len(labels) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]

        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_dir):
        ids = []
        for xml_path in glob.glob(str(image_sets_dir) + '/*.xml'):
            filename = os.path.basename(xml_path)
            ids.append(filename.split('.')[0])

        return ids

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)
    
    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        
        if self.transform:
            image, _ = self.transform(image)

        return image

    def _get_annotation(self, image_id):
        """
        all objects in one image
        
        Return:
            boxes  -> (num_objs, 4) np.float32
            labels -> (num_objs, )  np.int64
        """
        annotation_file = self.image_sets_dir / f'{image_id}.xml'
        objects = ET.parse(annotation_file).findall('object')
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            if class_name in self.class_dict:
                bbox = obj.find('bndbox')

                x0 = float(bbox.find('xmin').text) - 1
                y0 = float(bbox.find('ymin').text) - 1
                x1 = float(bbox.find('xmax').text) - 1
                y1 = float(bbox.find('ymax').text) - 1
                boxes.append([x0, y0, x1, y1])

                labels.append(self.class_dict[class_name])

                is_difficult_str = obj.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.image_sets_dir / f'{image_id}.jpg'
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
