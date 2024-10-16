import cv2
import numpy as np
from tqdm import tqdm

from hog import HOG_Parameters, hog

def grayscale_transform(X):
    return np.array([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in tqdm(X)])

def hog_transform(X, hog_parameters: HOG_Parameters):
    return np.array([hog(img,hog_parameters) for img in tqdm(X)])