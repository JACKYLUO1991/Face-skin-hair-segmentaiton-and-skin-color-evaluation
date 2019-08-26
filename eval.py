from __future__ import print_function, division

from keras.models import load_model
import numpy as np
import time
import cv2 as cv
import os
import argparse
from model.hrnet import HRNet
from model.hlrnet import HLRNet
from segmentation_models.backbones import get_preprocessing
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input as pinput
from keras import backend as K
from metric import *
import glob
import time
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


IMG_SIZE = None


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


def overlay(org_image, mask):
    """Display face and hair region"""
    assert org_image.shape[:2] == mask.shape[:2]
    face = mask[:, :, 1]
    hair = mask[:, :, 2]
    face_img = cv.bitwise_and(org_image, org_image, mask=face)
    hair_img = cv.bitwise_and(org_image, org_image, mask=hair)

    return face_img, hair_img


def overlay_single(org_image, mask, w1=1, w2=1):
    assert org_image.shape[:2] == mask.shape[:2]
    mask_n = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask_n[:, :, 0] = 1
    mask_n[:, :, 0] *= mask
    image = org_image * w1 + mask_n * w2

    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", '-is',
                        help="size of image", type=int, default=224)
    parser.add_argument("--backbone", '-bb',
                        help="backbone of image", type=str, default='seresnet18')
    parser.add_argument("--model_path", '-mp',
                        help="the path of model", type=str,
                        default='./checkpoints/CelebA/HLNet/model-222-0.159.h5')
    parser.add_argument("--class_number", '-cn',
                        help="number of output", type=int, default=3)
    parser.add_argument("--w1",
                        help="w1 weight", type=float, default=0.7)
    parser.add_argument("--w2",
                        help="w2 weight", type=float, default=0.3)
    parser.add_argument("--margin",
                        help="margin of image", type=float, default=0.4)
    parser.add_argument('--use_design', action='store_false')
    args = parser.parse_args()

    IMG_SIZE = args.image_size
    MODEL_PATH = args.model_path
    CLS_NUM = args.class_number
    BACKBONE = args.backbone
    USE_DESIGN = args.use_design
    W1 = args.w1
    W2 = args.w2

    model = load_model(MODEL_PATH, custom_objects={'mean_accuracy': mean_accuracy,
                                                   'mean_iou': mean_iou,
                                                   'frequency_weighted_iou': frequency_weighted_iou,
                                                   'pixel_accuracy': pixel_accuracy})
    if not os.path.exists("./demo/hair/hair_guided"):
        os.makedirs("./demo/hair/hair_guided")
    if not os.path.exists("./demo/hair/hair_colored"):
        os.makedirs("./demo/hair/hair_colored")
    if not os.path.exists("./demo/hair/colored"):
        os.makedirs("./demo/hair/colored")
    if not os.path.exists("./demo/hair/hair_mask"):
        os.makedirs("./demo/hair/hair_mask")

    for img_path in glob.glob("./demo/hair/Testing/*"):
        img_basename = os.path.basename(img_path)
        name = os.path.splitext(img_basename)[0]

        org_img = cv.imread(img_path)
        try:
            i_h, i_w, _ = org_img.shape
        except:
            raise ValueError("Reading image error...")

        img_resize = cv.resize(org_img, (IMG_SIZE, IMG_SIZE))
        img = img_resize[np.newaxis, :]

        if USE_DESIGN:
            img = pinput(img)
        else:
            preprocess_input = get_preprocessing(BACKBONE)
            img = preprocess_input(img)

        s = time.time()
        result_map = model.predict(img)[0]
        e = time.time() - s
        print("[info] inference time is %.4f s" % e)

        mask = _result_map_toimg(result_map)
        mask = cv.resize(mask, (i_w, i_h))[:, :, -1]

        cv.imwrite("./demo/hair/hair_mask/{}.jpg".format(name), mask)
        
        '''# Face and Hair channels
        mask_face = mask[:, :, 1]
        mask_hair = mask[:, :, -1]

        mask_guided_hair = cv.ximgproc.guidedFilter(
            guide=org_img, src=mask_hair, radius=4, eps=50, dDepth=-1)
        cv.imwrite(
            "./demo/hair/hair_guided/{}.jpg".format(name), mask_guided_hair)

        colored_hair = overlay_single(
            org_img, mask_guided_hair, w1=0.8, w2=0.2)
        cv.imwrite("./demo/hair/hair_colored/{}.jpg".format(name),
                    colored_hair)
        cv.imwrite("./demo/hair/colored/{}.jpg".format(name), mask)'''
