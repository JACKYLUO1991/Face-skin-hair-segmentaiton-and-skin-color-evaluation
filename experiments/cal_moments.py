# https://www.cnblogs.com/klchang/p/6512310.html
from __future__ import print_function, division

import cv2 as cv
import numpy as np
import tqdm
import time
import os
import sys
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from utils import *
from imutils import paths


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


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    CLASSES = 5

    images_list = []
    masks_list = []
    features_list = []
    classes_list = []

    s1 = time.time()
    for i in range(0, CLASSES):
        for imgpath in sorted(paths.list_images(str(i))):
            if os.path.splitext(imgpath)[-1] == '.jpg':
                images_list.append(imgpath)
                classes_list.append(int(i))
            elif os.path.splitext(imgpath)[-1] == '.png':
                masks_list.append(imgpath)
            else:
                raise ValueError("type error...")
    s2 = time.time()
    logger.info(f"Time use: {s2 - s1} s")

    for image_path, mask_path in tqdm.tqdm(zip(images_list, masks_list)):
        image = cv.imread(image_path)
        mask = cv.imread(mask_path, 0)
        features = color_moments(image, mask, color_space='ycrcb')
        features_list.append(features)

    logger.info(f"Time use: {time.time() - s2} s")
    logger.info("Data process ready...")

    # Resampling
    sm = SMOTE(sampling_strategy='all', random_state=2019)
    features_list, classes_list = sm.fit_resample(features_list, classes_list)

    X_train, X_test, y_train, y_test = train_test_split(
        features_list, classes_list, test_size=0.2, random_state=2019)

    clf = RandomForestClassifier(n_estimators=180, random_state=2019)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    joblib.dump(clf, 'skinColor.pkl')

    classify_report = classification_report(y_test, y_pred)
    logger.info('\n' + classify_report)

    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=['0', '1',
                                                   '2', '3', '4'], title='Confusion matrix')
    plt.show()
