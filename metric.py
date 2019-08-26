import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.utils.generic_utils import get_custom_objects


SMOOTH = K.epsilon()


def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:
    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
    Returns:
        IoU/Jaccard score in range [0, 1]
    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index
    """
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
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:
    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}
    The formula in terms of *Type I* and *Type II* errors:
    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}
    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
    Returns:
        F-score in range [0, 1]
    """
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
def mean_iou(y_true, y_pred, cls_num=3):
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
        result += nii/(ti - nii + loc_sum)
    return (1/nc) * result


def mean_accuracy(y_true, y_pred, cls_num=3):
    result = 0
    nc = tf.cast(tf.shape(y_true)[-1], tf.float32)
    for i in range(cls_num):
        nii = tf.reduce_sum(tf.round(tf.multiply(
            y_true[:, :, :, i], y_pred[:, :, :, i])))
        ti = tf.reduce_sum(y_true[:, :, :, i])
        if ti != 0:
            result += (nii/ti)
    return (1/nc) * result


def frequency_weighted_iou(y_true, y_pred, cls_num=3):
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
        result += (loc_sum * nii)/(ti - nii + loc_sum)
    sum_ti = tf.reduce_sum(y_true[:, :, :, :])
    return (1/sum_ti) * result


def pixel_accuracy(y_true, y_pred):
    # nii = number of pixels of classe i predicted to belong to class i
    sum_nii = tf.reduce_sum(tf.round(tf.multiply(
        y_true[:, :, :, :], y_pred[:, :, :, :])))
    # ti = number of pixels of class i
    sum_ti = tf.reduce_sum(y_true[:, :, :, :])
    return sum_nii/sum_ti


get_custom_objects().update({
    'pixel_accuracy': pixel_accuracy,
    'frequency_weighted_iou': frequency_weighted_iou,
    'mean_accuracy': mean_accuracy,
    'mean_iou': mean_iou
})
