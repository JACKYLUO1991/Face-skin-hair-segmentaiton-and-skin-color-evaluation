import numpy as np
import cv2
import os
import random
import glob
from keras.preprocessing.image import ImageDataGenerator
from segmentation_models.backbones import get_preprocessing
from keras.utils import Sequence
from keras.applications.imagenet_utils import preprocess_input as pinput
from albumentations import *


class HairGenerator(Sequence):
    '''
    1. 头发数据集迭代器, 每一个 Sequence 必须实现 __getitem__ 和 __len__ 方法
    2. 包含基本的数据扩充功能
    3. 添加了数据预处模块，方便后面的网络输入
    '''

    def __init__(self,
                 transformer,
                 root_dir='./data/CelebA',
                 mode='Training',
                 nb_classes=3,
                 batch_size=4,
                 backbone=None,
                 shuffle=False):

        assert mode in ['Training', 'Testing'], "数据集选择错误..."

        self.image_path_list = sorted(
            glob.glob(os.path.join(root_dir, 'Original', mode, '*')))
        self.mask_path_list = sorted(
            glob.glob(os.path.join(root_dir, 'GT', mode, '*')))
        self.transformer = transformer
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.shuffle = shuffle
        self.mode = mode
        self.backbone = backbone

    def __getitem__(self, idx):
        '''迭代数据'''
        images, masks = [], []

        for (image_path, mask_path) in zip(self.image_path_list[idx * self.batch_size: (idx+1) * self.batch_size],
                                           self.mask_path_list[idx * self.batch_size: (idx+1) * self.batch_size]):
            image = cv2.imread(image_path, 1)
            mask = cv2.imread(mask_path, 0)

            # 对图像和掩模进行长宽一致处理
            image = self._padding(image)
            mask = self._padding(mask)

            # 对应数据进行扩充整理
            augmentation = self.transformer(image=image, mask=mask)
            image = augmentation['image']
            mask = self._get_result_map(augmentation['mask'])

            images.append(image)
            masks.append(mask)

        images = np.array(images)
        masks = np.array(masks)

        # 根据不同的网络框架选择不同的数据预处理方式
        if self.backbone is not None:
            preprocess_input = get_preprocessing(self.backbone)
            images = preprocess_input(images)

        # 对作者设计的框架进行数据去均值处理
        else:
            images = pinput(images)

        return images, masks

    def __len__(self):
        '''每个epoch需要的步数'''
        return len(self.image_path_list) // self.batch_size

    def _padding(self, image):
        shape = image.shape
        h, w = shape[:2]
        width = np.max([h, w])
        padd_h = (width - h) // 2
        padd_w = (width - w) // 2
        if len(shape) == 3:
            padd_tuple = ((padd_h, width-h-padd_h),
                          (padd_w, width-w-padd_w), (0, 0))
        else:
            padd_tuple = ((padd_h, width-h-padd_h), (padd_w, width-w-padd_w))
        image = np.pad(image, padd_tuple, 'constant')
        return image

    def on_epoch_end(self):
        '''打乱图像顺序'''
        if self.shuffle:
            c = list(zip(self.image_path_list, self.mask_path_list))
            random.shuffle(c)
            self.image_path_list, self.mask_path_list = zip(*c)

    def _get_result_map(self, mask):
        '''
        对mask数据进行处理
        '''
        # mask.shape[0]: 代表图像的行
        # mask.shape[1]: 代表图像的列
        result_map = np.zeros((mask.shape[1], mask.shape[0], self.nb_classes))
        # For np.where calculation.
        # 0 (background pixel), 128 (face area pixel) or 255 (hair area pixel).
        skin = (mask == 128)
        hair = (mask == 255)
        background = np.logical_not(hair + skin)
        result_map[:, :, 0] = np.where(background, 1, 0)
        result_map[:, :, 1] = np.where(skin, 1, 0)
        result_map[:, :, 2] = np.where(hair, 1, 0)

        return result_map
