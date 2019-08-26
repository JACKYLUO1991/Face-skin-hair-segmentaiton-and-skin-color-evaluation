
# 参考资料:
# https://www.cnblogs.com/maybe2030/p/4585705.html
# https://blog.csdn.net/zhu_hongji/article/details/80443585
# https://blog.csdn.net/wsp_1138886114/article/details/80660014
# https://blog.csdn.net/gfjjggg/article/details/87919658
# https://baike.baidu.com/item/%E9%A2%9C%E8%89%B2%E7%9F%A9/19426187?fr=aladdin
# https://blog.csdn.net/langyuewu/article/details/4144139
from __future__ import print_function, division

from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from imutils import paths
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from utils import *


class RGBHistogram(Histogram):
    '''RGB Histogram'''

    def __init__(self, bins):
        super().__init__(bins)

    def describe(self, image, mask):
        hist_b = cv.calcHist([image], [0], mask, self.bins,
                             [0, 256])
        hist_g = cv.calcHist([image], [1], mask, self.bins,
                             [0, 256])
        hist_r = cv.calcHist([image], [2], mask, self.bins,
                             [0, 256])
        hist_b = hist_b / np.sum(hist_b)
        hist_g = hist_g / np.sum(hist_g)
        hist_r = hist_r / np.sum(hist_r)
        
        # 24 dimensions
        return np.concatenate([hist_b, hist_g, hist_r], axis=0).reshape(-1)



class HSVHistogram(Histogram):
    '''HSV Histogram'''

    def __init__(self, bins):
        super().__init__(bins)

    def describe(self, image, mask):
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hist_h = cv.calcHist([image], [0], mask, self.bins,
                             [0, 180])
        hist_s = cv.calcHist([image], [1], mask, self.bins,
                             [0, 256])
        hist_v = cv.calcHist([image], [2], mask, self.bins,
                             [0, 256])
        hist_h = hist_h / np.sum(hist_h)
        hist_s = hist_s / np.sum(hist_s)
        hist_v = hist_v / np.sum(hist_v)

        # 24 dimensions
        return np.concatenate([hist_h, hist_s, hist_v], axis=0).reshape(-1)


class YCrCbHistogram(Histogram):
    '''YCrCb Histogram'''

    def __init__(self, bins):
        super().__init__(bins)

    def describe(self, image, mask):
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        hist_y = cv.calcHist([image], [0], mask, self.bins,
                             [0, 256])
        hist_cr = cv.calcHist([image], [1], mask, self.bins,
                              [0, 256])
        hist_cb = cv.calcHist([image], [2], mask, self.bins,
                              [0, 256])
        hist_y = hist_y / np.sum(hist_y)
        hist_cr = hist_cr / np.sum(hist_cr)
        hist_cb = hist_cb / np.sum(hist_cb)

        # 24 dimensions
        return np.concatenate([hist_y, hist_cr, hist_cb], axis=0).reshape(-1)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    CLASSES = 5

    images_list = []
    masks_list = []
    features_list = []
    classes_list = []

    hist = YCrCbHistogram([8])

    s1 = time.time()
    for i in range(0, CLASSES):
        for imgpath in sorted(paths.list_images(str(i))):
            if os.path.splitext(imgpath)[-1] == '.jpg':
                images_list.append(imgpath)
                classes_list.append(i)
            elif os.path.splitext(imgpath)[-1] == '.png':
                masks_list.append(imgpath)
            else:
                raise ValueError("type error...")
    s2 = time.time()
    logger.info(f"Time use: {s2 - s1} s")

    for image_path, mask_path in zip(images_list, masks_list):
        # print(image_path, mask_path)
        image = cv.imread(image_path)
        mask = cv.imread(mask_path, 0)
        features = hist.describe(image, mask)
        # print(features)
        features_list.append(features)

    logger.info(f"Time use: {time.time() - s2} s")
    logger.info("Data process ready...")

    # Resampling
    sm = SMOTE(sampling_strategy='all', random_state=2019)
    features_list, classes_list = sm.fit_resample(features_list, classes_list)

    # Machine learning algorithm
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                     hidden_layer_sizes=(8, ), random_state=2019)
    clf = RandomForestClassifier(n_estimators=180, random_state=2019)
    # kf = KFold(n_splits=CLASSES, random_state=2019, shuffle=True).\
    #     get_n_splits(features_list)
    # scores = cross_val_score(clf, features_list, classes_list,
    #                          scoring='accuracy', cv=kf)
    # score = scores.mean()
    # logger.info(f"KFold score: {score}")

    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features_list, classes_list, test_size=0.2, random_state=2019)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    classify_report = classification_report(y_test, y_pred)
    logger.info('\n' + classify_report)

    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=['0', '1',
                                                   '2', '3', '4'], title='Confusion matrix')
    plt.show()

    # Save model
    # https://blog.csdn.net/qiang12qiang12/article/details/81001839
    # How to load model: 
    #   1. clf = joblib.load('models/histogram.pkl')
    #   2. clf.predict(X_test)

    # joblib.dump(clf, 'models/histogram.pkl')
