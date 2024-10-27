
import re
import cv2
import os, shutil
import random
import json
from dataset import get_dataset_path
from parameters import HOG_Parameters, SVM_Parameters, iterate_model_parameters
from dataset import datasets
from transform import grayscale_transform, hog_transform
import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

valid_objects = [
    'person',
    'people'
]

def move_files_to_parent(folder_path):
    for root, dirs, files in os.walk(folder_path):
        if len(dirs) == 1:
            child_folder = dirs[0]
            child_folder_path = os.path.join(root, child_folder)
            for file_name in os.listdir(child_folder_path):
                file_path = os.path.join(child_folder_path, file_name)
                if os.path.isfile(file_path):
                    shutil.move(file_path, root)
                    print(f"Moved: {file_path} -> {root}")
            os.rmdir(child_folder_path)
            print(f"Removed empty folder: {child_folder_path}")

def separate_files(src_folder):
    annotations_folder = os.path.join(src_folder, 'annotations/set00')
    frames_folder = os.path.join(src_folder, 'frame/set00')

    os.makedirs(annotations_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)

    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)

        if os.path.isdir(file_path):
            continue

        if filename.endswith('.xml'):
            shutil.move(file_path, annotations_folder)
            print(f"Moved {filename} to {annotations_folder}")

        elif filename.endswith('.jpg'):
            shutil.move(file_path, frames_folder)
            print(f"Moved {filename} to {frames_folder}")

def draw_bounding_boxes(image_path, bbox_list):
    image = cv2.imread(image_path)
    for (xmin, ymin, xmax, ymax) in bbox_list:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def retain_30th_frame():
    root_dir = r'/Users/adamsam/repos/ee/Pedestrian-Detection/datasets/caltech_30/Test'
    annotation_dir = os.path.join(root_dir, 'annotations')
    frame_dir = os.path.join(root_dir, 'frame')
    frame_instance = 0
    for frame_subdir in tqdm(os.listdir(frame_dir)):
        frame_subdir_path = os.path.join(frame_dir, frame_subdir)
        if(os.path.isdir(frame_subdir_path)):
            frame_files = os.listdir(os.path.join(frame_subdir_path, 'frame'))
            for frame_file in frame_files:
                file_location = os.path.join(frame_subdir_path, 'frame', frame_file)

                if not os.path.isfile(file_location):
                    continue

                frame_instance += 1

                if frame_instance % 30 != 0:
                    os.remove(file_location)
                    annotation_file_location = os.path.join(annotation_dir, frame_subdir, 'bbox', frame_file.split('.')[0] + '.xml')
                    if os.path.isfile(annotation_file_location):
                        os.remove(annotation_file_location)

def get_concatenated_test_samples(svm_parameters: SVM_Parameters):
    total_x = []
    total_y = []
    for dataset in datasets:
        X_test_raw = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset))
        X_test = hog_transform(
            grayscale_transform(X_test_raw),
            svm_parameters.hog_parameters
        )
        y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', dataset))

        total_x.append(X_test)
        total_y.append(y_test)
    return np.concatenate(total_x), np.concatenate(total_y)

def get_attr_names(cls):
    if not hasattr(cls, '__class__'):
        raise ValueError("The provided object is not an instance of a class.")
    return cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]

def get_detector_pairs(prop_name, prop_value):
    detectors_with_prop, detectors_without_prop = get_detectors_by_prop(prop_name, prop_value, return_inadequate=True)
    print(detectors_with_prop[0].hog_parameters.get_hog_name())
    print(detectors_without_prop[0].hog_parameters.get_hog_name())
    zipped_detectors = []
    for detector_with_prop in detectors_with_prop:
        for detector_without_prop in detectors_without_prop:
            identical_features = True
            for attr_name in get_attr_names(detector_with_prop):
                if attr_name == prop_name:
                    continue
                if getattr(detector_with_prop, attr_name) != getattr(detector_without_prop, attr_name):
                    identical_features = False
                    break
            if identical_features:
                zipped_detectors.append((detector_with_prop, detector_without_prop))
    return zipped_detectors
                    
    
    

def get_detectors_by_prop(prop, prop_value, return_inadequate=False):
    if prop not in get_attr_names(SVM_Parameters) and prop not in get_attr_names(HOG_Parameters):
        raise ValueError(f"Property {prop} not found in SVM_Parameters or HOG_Parameters.")
    
    detectors = []
    inadequate_detectors = []
    for svm_params in iterate_model_parameters():
        if prop == 'window_size':
            if getattr(svm_params, prop) == prop_value:
                detectors.append(svm_params)
        elif getattr(svm_params.hog_parameters, prop) == prop_value:
            detectors.append(svm_params)
        else:
            inadequate_detectors.append(svm_params)
            
    if return_inadequate:
        return detectors, inadequate_detectors
    return detectors

def extract_svm_params(detector_name):
    orientations = re.search(r'orientations_(\d+)', detector_name)
    pixels_per_cell = re.search(r'pixels_per_cell_\((\d+),\s*(\d+)\)', detector_name)
    cells_per_block = re.search(r'cells_per_block_\((\d+),\s*(\d+)\)', detector_name)
    block_stride = re.search(r'block_stride_\((\d+),\s*(\d+)\)', detector_name)
    holistic_derivative_mask = re.search(r'holistic_derivative_mask_(True|False)', detector_name)
    window_size = re.search(r'window_\((\d+),\s*(\d+)\)', detector_name)

    # Extract the values
    W_h, W_w = window_size.group(1), window_size.group(2) if window_size else (None, None)
    orientations_value = orientations.group(1) if orientations else None
    c_h, c_w = pixels_per_cell.group(1), pixels_per_cell.group(2) if pixels_per_cell else (None, None)
    b_h, b_w = cells_per_block.group(1), cells_per_block.group(2) if cells_per_block else (None, None)
    s_h, s_w = block_stride.group(1), block_stride.group(2) if block_stride else (None, None)
    hdm = holistic_derivative_mask.group(1) if holistic_derivative_mask else None
    
    return W_h, W_w, orientations_value, c_h, c_w, b_h, b_w, s_h, s_w, hdm

def get_svm_params(detector_name):
    W_h, W_w, orientations_value, c_h, c_w, b_h, b_w, s_h, s_w, hdm = extract_svm_params(detector_name)
    
    hog_parameters = HOG_Parameters(
        orientations=int(orientations_value),
        pixels_per_cell=(int(c_h), int(c_w)),
        cells_per_block=(int(b_h), int(b_w)),
        block_stride=(int(s_h), int(s_w)),
        holistic_derivative_mask=True if hdm == 'True' else False,
        block_norm='L2-Hys'
    )
    return SVM_Parameters(
        hog_parameters=hog_parameters,
        window_size=(int(W_h), int(W_w))
    )

def get_short_svm_name(detector_name, with_window_size=False):
    W_h, W_w, orientations_value, c_h, c_w, b_h, b_w, s_h, s_w, hdm = extract_svm_params(detector_name)
    
    name = f"{orientations_value}-({c_h}, {c_w})-({b_h}, {b_w})-({s_h}, {s_w})-{hdm}"
    
    if with_window_size:
        name = f"({W_h}, {W_w})-" + name
    return name