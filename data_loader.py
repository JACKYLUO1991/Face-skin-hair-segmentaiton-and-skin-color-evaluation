import numpy as np
import cv2
import os
import random
import glob

from keras.utils import Sequence
from keras.applications.imagenet_utils import preprocess_input as pinput


class HairGenerator(Sequence):

    def __init__(self,
                 transformer,
                 root_dir,
                 mode='Training',
                 nb_classes=3,
                 batch_size=4,
                 backbone=None,
                 shuffle=False):

        # backbone fit for segmentation_models，have been deleted now...
        assert mode in ['Training', 'Testing'], "Data set selection error..."

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
        images, masks = [], []

        for (image_path, mask_path) in zip(self.image_path_list[idx * self.batch_size: (idx + 1) * self.batch_size],
                                           self.mask_path_list[idx * self.batch_size: (idx + 1) * self.batch_size]):
            image = cv2.imread(image_path, 1)
            mask = cv2.imread(mask_path, 0)

            image = self._padding(image)
            mask = self._padding(mask)

            # augumentation
            augmentation = self.transformer(image=image, mask=mask)
            image = augmentation['image']
            mask = self._get_result_map(augmentation['mask'])

            images.append(image)
            masks.append(mask)

        images = np.array(images)
        masks = np.array(masks)
        images = pinput(images)

        return images, masks

    def __len__(self):
        """Steps required per epoch"""
        return len(self.image_path_list) // self.batch_size

    def _padding(self, image):
        shape = image.shape
        h, w = shape[:2]
        width = np.max([h, w])
        padd_h = (width - h) // 2
        padd_w = (width - w) // 2
        if len(shape) == 3:
            padd_tuple = ((padd_h, width - h - padd_h),
                          (padd_w, width - w - padd_w), (0, 0))
        else:
            padd_tuple = ((padd_h, width - h - padd_h), (padd_w, width - w - padd_w))
        image = np.pad(image, padd_tuple, 'constant')
        return image

    def on_epoch_end(self):
        """Shuffle image order"""
        if self.shuffle:
            c = list(zip(self.image_path_list, self.mask_path_list))
            random.shuffle(c)
            self.image_path_list, self.mask_path_list = zip(*c)

    def _get_result_map(self, mask):
        """Processing mask data"""

        # mask.shape[0]: row, mask.shape[1]: column
        result_map = np.zeros((mask.shape[1], mask.shape[0], self.nb_classes))
        # 0 (background pixel), 128 (face area pixel) or 255 (hair area pixel).
        skin = (mask == 128)
        hair = (mask == 255)

        if self.nb_classes == 2:
            # hair = (mask > 128)
            background = np.logical_not(hair)
            result_map[:, :, 0] = np.where(background, 1, 0)
            result_map[:, :, 1] = np.where(hair, 1, 0)
        elif self.nb_classes == 3:
            background = np.logical_not(hair + skin)
            result_map[:, :, 0] = np.where(background, 1, 0)
            result_map[:, :, 1] = np.where(skin, 1, 0)
            result_map[:, :, 2] = np.where(hair, 1, 0)
        else:
            raise Exception("error...")

        return result_map
