from __future__ import print_function, division

from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input as pinput

import cv2 as cv
import numpy as np
import os
import argparse
from metric import *
import glob
from model.fast_scnn import resize_image
from segmentation_models.losses import *

import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

IMG_SIZE = None


def vis_parsing_maps(im, parsing_anno, data_name):
    part_colors = [[255, 255, 255], [0, 255, 0], [255, 0, 0]]

    if data_name == 'figaro1k':
        part_colors = [[255, 255, 255], [255, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros(
        (parsing_anno.shape[0], parsing_anno.shape[1], 3))

    for pi in range(len(part_colors)):
        index = np.where(parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    # Guided filter
    # vis_parsing_anno_color = cv.ximgproc.guidedFilter(
    #     guide=vis_im, src=vis_parsing_anno_color, radius=4, eps=50, dDepth=-1)
    vis_im = cv.addWeighted(vis_im, 0.7, vis_parsing_anno_color, 0.3, 0)

    return vis_im


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size",
                        help="size of image", type=int, default=256)
    parser.add_argument("--model_path",
                        help="the path of model", type=str,
                        default='./weights/celebhair/exper/fastscnn/model.h5')
    args = parser.parse_args()

    IMG_SIZE = args.image_size
    MODEL_PATH = args.model_path

    if MODEL_PATH.split('/')[-2] == 'lednet':
        from model.lednet import LEDNet

        model = LEDNet(2, 3, (256, 256, 3)).model()
        model.load_weights(MODEL_PATH)

    else:
        model = load_model(MODEL_PATH, custom_objects={'mean_accuracy': mean_accuracy,
                                                       'mean_iou': mean_iou,
                                                       'frequency_weighted_iou': frequency_weighted_iou,
                                                       'pixel_accuracy': pixel_accuracy,
                                                       'categorical_crossentropy_plus_dice_loss': cce_dice_loss,
                                                       'resize_image': resize_image})

    data_name = MODEL_PATH.split('/')[2]

    for img_path in glob.glob(os.path.join("./demo", data_name, "*.jpg")):
        img_basename = os.path.basename(img_path)
        name = os.path.splitext(img_basename)[0]

        org_img = cv.imread(img_path)
        try:
            h, w, _ = org_img.shape
        except:
            raise IOError("Reading image error...")

        img_resize = cv.resize(org_img, (IMG_SIZE, IMG_SIZE))
        img = img_resize[np.newaxis, :]
        # pre-processing
        img = pinput(img)

        result_map = np.argmax(model.predict(img)[0], axis=-1)
        out = vis_parsing_maps(img_resize, result_map, data_name)
        out = cv.resize(out, (w, h), interpolation=cv.INTER_NEAREST)

        cv.imwrite(os.path.join("./demo", data_name, "{}.png").format(name), out)
