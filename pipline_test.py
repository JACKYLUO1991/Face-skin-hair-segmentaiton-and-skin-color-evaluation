from __future__ import print_function, division

from keras.models import load_model
import numpy as np
import time
import cv2 as cv
import os
import sys
import argparse
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input as pinput
from keras import backend as K

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from segmentation_models.backbones import get_preprocessing
from model.hlnet import HLRNet
from model.hrnet import HRNet
from segmentation_models import PSPNet, Unet, FPN, Linknet
from mtcnn.mtcnn import MTCNN
from metric import *
from imutils import paths

IMG_SIZE = None


def color_moments(image, mask, color_space):
    """
    function: Color Moment Features
    image: raw image
    mask: image mask
    color_space: 'rgb' or 'lab' or 'ycrcb' or 'hsv'
    """
    assert image.shape[:2] == mask.shape
    assert color_space.lower() in ['lab', 'rgb', 'ycrcb', 'hsv']

    if color_space.lower() == 'rgb':
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    elif color_space.lower() == 'hsv':
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    elif color_space.lower() == 'lab':
        image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    elif color_space.lower() == 'ycrcb':
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    else:
        raise ValueError("Color space error...")

    # Split image channels info
    c1, c2, c3 = cv.split(image)
    color_feature = []

    # Only process mask != 0 channel region
    c1 = c1[np.where(mask != 0)]
    c2 = c2[np.where(mask != 0)]
    c3 = c3[np.where(mask != 0)]

    # Extract mean
    mean_1 = np.mean(c1)
    mean_2 = np.mean(c2)
    mean_3 = np.mean(c3)

    # Extract variance
    variance_1 = np.std(c1)
    variance_2 = np.std(c2)
    variance_3 = np.std(c3)

    # Extract skewness
    skewness_1 = np.mean(np.abs(c1 - mean_1) ** 3) ** (1. / 3)
    skewness_2 = np.mean(np.abs(c1 - mean_2) ** 3) ** (1. / 3)
    skewness_3 = np.mean(np.abs(c1 - mean_3) ** 3) ** (1. / 3)

    color_feature.extend(
        [mean_1, mean_2, mean_3, variance_1, variance_2,
            variance_3, skewness_1, skewness_2, skewness_3])

    return color_feature


def _result_map_toimg(result_map):
    '''show result map'''
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    argmax_id = np.argmax(result_map, axis=-1)
    background = (argmax_id == 0)
    skin = (argmax_id == 1)
    hair = (argmax_id == 2)

    img[:, :, 0] = np.where(background, 255, 0)
    img[:, :, 1] = np.where(skin, 255, 0)
    img[:, :, 2] = np.where(hair, 255, 0)

    return img


def imcrop(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0), cv.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", '-is',
                        help="size of image", type=int, default=224)
    parser.add_argument("--backbone", '-bb',
                        help="backbone of image", type=str, default='seresnet18')
    parser.add_argument("--model_path", '-mp',
                        help="the path of model", type=str,
                        default='./checkpoints/CelebA/HLNet/model-222-0.159.h5')
    parser.add_argument("--margin",
                        help="margin of image", type=float, default=0.3)
    parser.add_argument('--use_design', action='store_false')
    args = parser.parse_args()

    IMG_SIZE = args.image_size
    MODEL_PATH = args.model_path
    BACKBONE = args.backbone
    USE_DESIGN = args.use_design

    detector = MTCNN()
    clf = joblib.load('./experiments/skinGrade/skinColor.pkl')
    model = load_model(MODEL_PATH, custom_objects={'mean_accuracy': mean_accuracy,
                                                   'mean_iou': mean_iou,
                                                   'frequency_weighted_iou': frequency_weighted_iou,
                                                   'pixel_accuracy': pixel_accuracy})
    colorHue = ['Ivory white', 'Porcelain white',
                'natural color', 'Yellowish', 'Black']

    for img_path in paths.list_images("./data/Testing"):
        t = time.time()

        org_img = cv.imread(img_path)
        try:
            org_img.shape
        except:
            raise ValueError("Reading image error...")

        org_img_rgb = org_img[:, :, ::-1] # RGB
        detected = detector.detect_faces(org_img_rgb)

        if len(detected) != 1:
            print("[INFO] multi faces or no face...")
            continue

        d = detected[0]['box']
        x1, y1, x2, y2, w, h = d[0], d[1], d[0] + d[2], d[1] + d[3], d[2], d[3]
        xw1 = int(x1 - args.margin * w)
        yw1 = int(y1 - args.margin * h)
        xw2 = int(x2 + args.margin * w)
        yw2 = int(y2 + args.margin * h)
        cropped_img = imcrop(org_img, xw1, yw1, xw2, yw2)
        o_h, o_w, _ = cropped_img.shape

        cropped_img_resize = cv.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
        img = cropped_img_resize[np.newaxis, :]


        # only subtract mean value
        img = pinput(img)

        result_map = model.predict(img)[0]
        mask = _result_map_toimg(result_map)
        mask = cv.resize(mask, (o_w, o_h))

        # Face channel
        mask_face = mask[:, :, 1]
        features = color_moments(cropped_img, mask_face, color_space='ycrcb')
        features = np.array(features, np.float32)[np.newaxis, :]
        skinHue = colorHue[clf.predict(features)[0]]

        cv.rectangle(org_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.putText(org_img, 'Color: {}'.format(skinHue), (x1, y1+30),
                    cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        print(time.time() - t)  # testing time
        cv.imshow("image", org_img)
        cv.waitKey(-1)

