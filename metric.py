import keras.backend as K
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

CLS_NUM = 2  # should be modified according to class number

SMOOTH = K.epsilon()


def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou


def f1_score(gt, pr, class_weights=1, beta=1, smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score


def precision(gt, pr, class_weights=1, smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    tp = K.sum(gt * pr, axis=axes)
    score = tp / (K.sum(pr, axis=axes) + smooth)

    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score


def recall(gt, pr, class_weights=1, smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    tp = K.sum(gt * pr, axis=axes)
    score = tp / (K.sum(gt, axis=axes) + smooth)

    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score


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
