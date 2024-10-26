import os

import cv2
import numpy as np
from tqdm import tqdm

from hog import hog, m_hog
from parameters import SVM_Parameters, HOG_Parameters


def grayscale_transform(X):
    return np.array([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in tqdm(X)])

def hog_transform_cache(X, svm_parameters: SVM_Parameters, category, dataset):
    if category not in ['train', 'test']:
        raise ValueError('Category must be either train or test')
    if dataset not in ['INRIA', 'caltech_30', 'PnPLO']:
        raise ValueError('Dataset must be either INRIA, caltech_30, or PnPLO')

    def get_cached_hog_name():
        return f'../cache/{svm_parameters.get_svm_name()}_{category}_{dataset}.npy'

    if os.path.exists(get_cached_hog_name()):
        return np.load(get_cached_hog_name())
    else:
        X_hog = hog_transform(X, svm_parameters.hog_parameters)
        np.save(get_cached_hog_name(), X_hog)
        return X_hog

def hog_transform(X, hog_parameters: HOG_Parameters):
    return np.array([hog(img,hog_parameters) for img in tqdm(X)])

def m_hog_transform(X, hog_parameters):
    return np.array([m_hog(img, hog_parameters) for img in tqdm(X)])