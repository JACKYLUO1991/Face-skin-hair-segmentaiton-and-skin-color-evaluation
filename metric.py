import keras.backend as K
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

CLS_NUM = 2  # should be modified according to class number

SMOOTH = K.epsilon()

# https: // blog.csdn.net/majinlei121/article/details/78965435
def mean_iou(y_true, y_pred, cls_num=CLS_NUM):
    result = 0
    nc = tf.cast(tf.shape(y_true)[-1], tf.float32)
    for i in range(cls_num):
        # nii = number of pixels of classe i predicted to belong to class i
        nii = tf.reduce_sum(tf.round(tf.multiply(
            y_true[:, :, :, i], y_pred[:, :, :, i])))
        ti = tf.reduce_sum(y_true[:, :, :, i])  # number of pixels of class i
        loc_sum = 0
        for j in range(cls_num):
            # number of pixels of classe j predicted to belong to class i
            nji = tf.reduce_sum(tf.round(tf.multiply(
                y_true[:, :, :, j], y_pred[:, :, :, i])))
            loc_sum += nji
        result += nii / (ti - nii + loc_sum)
    return (1 / nc) * result


def mean_accuracy(y_true, y_pred, cls_num=CLS_NUM):
    result = 0
    nc = tf.cast(tf.shape(y_true)[-1], tf.float32)
    for i in range(cls_num):
        nii = tf.reduce_sum(tf.round(tf.multiply(
            y_true[:, :, :, i], y_pred[:, :, :, i])))
        ti = tf.reduce_sum(y_true[:, :, :, i])
        if ti != 0:
            result += (nii / ti)
    return (1 / nc) * result


def frequency_weighted_iou(y_true, y_pred, cls_num=CLS_NUM):
    result = 0
    for i in range(cls_num):
        nii = tf.reduce_sum(tf.round(tf.multiply(
            y_true[:, :, :, i], y_pred[:, :, :, i])))
        ti = tf.reduce_sum(y_true[:, :, :, i])
        loc_sum = 0
        for j in range(cls_num):
            nji = tf.reduce_sum(tf.round(tf.multiply(
                y_true[:, :, :, j], y_pred[:, :, :, i])))
            loc_sum += nji
        result += (loc_sum * nii) / (ti - nii + loc_sum)
    sum_ti = tf.reduce_sum(y_true[:, :, :, :])
    return (1 / sum_ti) * result


def pixel_accuracy(y_true, y_pred):
    # nii = number of pixels of classe i predicted to belong to class i
    sum_nii = tf.reduce_sum(tf.round(tf.multiply(
        y_true[:, :, :, :], y_pred[:, :, :, :])))
    # ti = number of pixels of class i
    sum_ti = tf.reduce_sum(y_true[:, :, :, :])
    return sum_nii / sum_ti


get_custom_objects().update({
    'pixel_accuracy': pixel_accuracy,
    'frequency_weighted_iou': frequency_weighted_iou,
    'mean_accuracy': mean_accuracy,
    'mean_iou': mean_iou
})
