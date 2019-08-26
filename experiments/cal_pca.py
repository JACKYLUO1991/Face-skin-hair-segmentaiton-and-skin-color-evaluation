# color-auto-correlogram
# https://blog.csdn.net/u013066730/article/details/53609859
from __future__ import print_function, division

import numpy as np
import cv2 as cv
import sys
import os
import tqdm
import time
import csv
import pandas as pd
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from utils import *
from imutils import paths


class RGBHistogram(Histogram):
    '''RGB Histogram'''

    def __init__(self, bins):
        super().__init__(bins)

    def describe(self, image, mask):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        hist = cv.calcHist([image], [0, 1, 2], mask,
                           self.bins, [0, 256, 0, 256, 0, 256])
        hist = hist / np.sum(hist)

        # 512 dimensions
        return hist.flatten()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    CLASSES = 5
    K_ClUSTER = 15

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

    hist = RGBHistogram([8, 8, 8])

    for image_path, mask_path in tqdm.tqdm(zip(images_list, masks_list)):
        image = cv.imread(image_path)
        mask = cv.imread(mask_path, 0)
        features = hist.describe(image, mask)
        features_list.append(features)

    logger.info(f"Time use: {time.time() - s2} s")
    logger.info("Data process ready...")
    
    assert len(features_list) == len(classes_list)

    # PCA Dimensionality Reduction
    pca = PCA(n_components=K_ClUSTER, random_state=2019)
    # pca.fit(features_list)
    # logger.info(pca.explained_variance_ratio_)
    newX = pca.fit_transform(features_list)

    X_train, X_test, y_train, y_test = train_test_split(
        newX, classes_list, test_size=0.2, random_state=2019)

    clf = RandomForestClassifier(n_estimators=180, random_state=2019)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    classify_report = classification_report(y_test, y_pred)
    logger.info('\n' + classify_report)

    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=['0', '1',
                                                   '2', '3', '4'], title='Confusion matrix')
    plt.show()
