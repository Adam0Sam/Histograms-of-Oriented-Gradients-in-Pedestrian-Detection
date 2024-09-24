import os

import cv2
import joblib
import numpy as np
from skimage.transform import pyramid_gaussian
from tqdm import tqdm

from debug import get_single_debug_image_name
from hog import HOG_Parameters, hog
from utils import prepare_data, get_grayscale_image, sliding_window, NMS
from sklearn.svm import SVC

class SVM_Parameters:
    def __init__(self, hog_parameters: HOG_Parameters, window_size, step_size,):
        self.hog_parameters = hog_parameters
        self.window_size = window_size
        self.step_size = step_size
    def get_svm_name(self):
        return "svm_" + self.hog_parameters.get_hog_name() + "_window_" + str(self.window_size) + "_step_" + str(self.step_size)

def train_svm(svm_parameters: SVM_Parameters, image_folder, force_train=False, save_model=True):
    model_dir = os.path.join("../", "saved_models")
    model_file_name = os.path.join(model_dir, svm_parameters.get_svm_name() + ".pkl")
    if not force_train:
        if os.path.exists(model_file_name):
            return joblib.load(model_file_name)

    training_data_points, training_labels = prepare_data(image_folder, svm_parameters.window_size)
    hog_feature_vectors = []

    print("Computing HoG features")
    for data_point in tqdm(training_data_points):
        data_point = get_grayscale_image(data_point)

        feature_vector = hog(
            data_point,
            svm_parameters.hog_parameters
        )

        hog_feature_vectors.append(feature_vector)

    classifier = SVC(kernel='linear')
    print("Training SVM Classifier")
    classifier.fit(np.array(hog_feature_vectors), np.array(training_labels))

    if save_model:
        joblib.dump(classifier, model_file_name, compress=1)
    return classifier

def get_model_predictions(svm_parameters: SVM_Parameters, image_folder):
    classifier = train_svm(svm_parameters, image_folder)
    image_dir = os.path.join(image_folder, "PNGImages")
    test_image_names = [file for file in os.listdir(image_dir) if file.startswith(get_single_debug_image_name())]
    SCALE_FACTOR = 1.25
    CONFIDENCE_THRESHOLD = 0.75
    results = {'test_images': [], 'detections': []}

    image_id = 0
    category_id = 1

    for image_name in tqdm(test_image_names):

        image_id += 1
        results['test_images'].append({
            "file_name": image_name,
            "image_id": image_id
        })

        image = cv2.imread(os.path.join(image_dir, image_name))

        original_image_height = image.shape[0]
        original_image_width = image.shape[1]

        # Resizing the image for speed purposes
        # Is that necessary?
        image = cv2.resize(image, (400, 256))
        resize_h_ratio = original_image_height / 256
        resize_w_ratio = original_image_width / 400

        image = get_grayscale_image(image)
        rects = []
        confidence = []

        scale = 0

        for scaled_image in pyramid_gaussian(image,downscale=SCALE_FACTOR):
            if(
                scaled_image.shape[0] < svm_parameters.window_size[0] and
                scaled_image.shape[1] < svm_parameters.window_size[1]
            ): break

            windows = sliding_window(scaled_image, svm_parameters.window_size, svm_parameters.step_size)
            for (x,y,window) in windows:
                # somethign def doesnt work here
                if window.shape[0] == svm_parameters.window_size[0] and window.shape[1] == svm_parameters.window_size[1]:
                    feature_vector = hog(
                        window,
                        svm_parameters.hog_parameters
                    )
                    feature_vector = feature_vector.reshape(1, -1) # or feature_vector = [feature_vector]
                    prediction = classifier.predict(feature_vector)
                    if prediction == 1:
                        confidence_score = classifier.decision_function(feature_vector)
                        if confidence_score > CONFIDENCE_THRESHOLD:
                            left_pos = int(x * (SCALE_FACTOR ** scale) * resize_w_ratio)
                            top_pos = int(y * (SCALE_FACTOR ** scale) * resize_h_ratio)

                            rects.append([
                                left_pos,
                                top_pos,
                                left_pos + original_image_width,
                                top_pos + original_image_height
                            ])
                            confidence.append([confidence_score])

            scale += 1

        rects,scores = NMS(rects,confidence)
        for rect,score in zip(rects,scores):
            x1,y1,x2,y2 = rect.tolist()
            results['detections'].append({"image_id":image_id,"category_id":category_id,"bbox":[x1,y1,x2-x1,y2-y1],"score":score.item()})

    return results


