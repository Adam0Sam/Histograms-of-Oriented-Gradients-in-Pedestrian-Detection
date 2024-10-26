import os
import joblib
import numpy as np

from hog import HOG_Parameters, hog
from parameters import SVM_Parameters
from transform import grayscale_transform, hog_transform
from sklearn.svm import SVC


def load_svm(svm_parameters: SVM_Parameters, model_dir=None, custom_name=None):
    model_name = custom_name if custom_name is not None else svm_parameters.get_svm_name()
    model_dir = model_dir if model_dir is not None else '../computed/models'
    model_file_name = os.path.join(model_dir, model_name + ".pkl")
    if os.path.exists(model_file_name):
        return joblib.load(model_file_name)
    raise Exception("Model not found")

def train_svm(svm_parameters: SVM_Parameters, data_points_location, labels_location, overwrite=False, custom_name=None, kernel_type="linear"):
    from sklearn.linear_model import SGDClassifier
    model_name = custom_name if custom_name is not None else svm_parameters.get_svm_name()

    model_file_path = os.path.join('../computed/models', model_name + ".pkl")

    if os.path.exists(model_file_path):
      if overwrite:
        print("Removing existing model")
        os.remove(model_file_path)
      else:
        print("Model already exists")
        model = load_svm(svm_parameters, model_dir='../computed/models', custom_name=custom_name)
        single_x = np.load(data_points_location)[0]
        single_x_gray = grayscale_transform(np.array([single_x]))
        single_x_hog = hog_transform(single_x_gray, svm_parameters.hog_parameters)
        try:
            model.predict(single_x_hog)
            print("Model loaded successfully")
            return
        except:
            print("Model failed to load, retraining")

    if os.path.exists(data_points_location) and os.path.exists(labels_location):
        training_data_points = np.load(data_points_location)
        training_labels = np.load(labels_location)
    else:
        raise Exception(
            "No data points or labels found",
            data_points_location,
            labels_location
        )

    x_train = np.load(data_points_location)
    y_train = np.load(labels_location)

    x_train_gray = grayscale_transform(x_train)
    x_train_hog = hog_transform(x_train_gray, svm_parameters.hog_parameters)

    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(x_train_hog, y_train)

    joblib.dump(sgd_clf, model_file_path)

