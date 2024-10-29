import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

datasets = ['INRIA', 'caltech_30', 'PnPLO']
dataset_name_map = {
    'INRIA': 'INRIA',
    'caltech_30': 'Caltech',
    'PnPLO': 'PnPLO'
}

class SampleCount:
    def __init__(self, pos_count, neg_count):
        self.pos = pos_count
        self.neg = neg_count

class LabeledDataSet:
    def __init__(self, points, labels, sample_count: SampleCount):
        self.points = points
        self.labels = labels
        self.sample_count = sample_count

def parse_pascal_voc_annotations(file_name):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_name)
    root = tree.getroot()
    bbox = []

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        bbox.append([
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)
        ])
    return bbox


def prepare_labeled_datasets(image_folder, window_size, test_size=0.2, random_state=42):
    """
    Notes:
    ------
        - The function expects the image folder to have a specific structure with subdirectories for frames and annotations.
        - Positive samples are created from bounding boxes in the annotations, and negative samples are created by randomly cropping windows from the images.
    """
    image_dir = os.path.join(image_folder, "frame")
    annotation_dir = os.path.join(image_folder, "annotations")

    image_subdirs = [
        os.path.join(image_dir, subdir)
        for subdir in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, subdir))
    ]
    images = [os.path.join(subdir, file) for subdir in image_subdirs for file in os.listdir(subdir) if
              os.path.isfile(os.path.join(subdir, file))]


    train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)

    def process_images(image_list):
        data_points = []
        labels = []
        num_pos = 0
        num_neg = 0

        for num, image_file_location in enumerate(tqdm(image_list)):
            image = cv2.imread(image_file_location)

            partial_location = image_file_location.split(os.sep)[-2:]
            annotation_file_location = os.path.join(
                annotation_dir,
                "/".join(map(str, partial_location))
            )[:-4]

            if os.path.exists(annotation_file_location + ".xml"):
                bbox_arr = parse_pascal_voc_annotations(annotation_file_location + ".xml")
            else:
                raise Exception(f"Annotation file {annotation_file_location} not found")

            for _ in range(3):
                h, w = image.shape[:2]

                if h > window_size[0] or w > window_size[1]:
                    h = h - window_size[0]
                    w = w - window_size[1]
                    overlap = [True for i in bbox_arr]
                    max_loop = 0
                    while np.any(overlap):
                        max_loop += 1
                        if max_loop == 10:
                            break
                        overlap = [True for i in bbox_arr]
                        x = random.randint(0, w)
                        y = random.randint(0, h)
                        window = [x, y, x + window_size[1], y + window_size[0]]
                        for var, bbox in enumerate(bbox_arr):
                            dx = min(bbox[2], window[2]) - max(bbox[0], window[0])
                            dy = min(bbox[3], window[3]) - max(bbox[1], window[1])
                            if dx <= 0 or dy <= 0:
                                overlap[var] = False
                    if max_loop < 10:
                        img = image[window[1]:window[3], window[0]:window[2]]
                        data_points.append(img)
                        labels.append(0)
                        num_neg += 1

            # Process positive samples (bounding boxes)
            for box in bbox_arr:
                img = image[box[1]:box[3], box[0]:box[2]]
                img_resized = cv2.resize(img, (window_size[1], window_size[0]))
                data_points.append(img_resized)
                labels.append(1)
                num_pos += 1

        return data_points, labels, num_pos, num_neg

    train_data, train_labels, train_pos, train_neg = process_images(train_images)
    test_data, test_labels, test_pos, test_neg = process_images(test_images)


    labeled_training_set = LabeledDataSet(np.array(train_data), np.array(train_labels), SampleCount(train_pos, train_neg))
    labeled_testing_set = LabeledDataSet(np.array(test_data), np.array(test_labels), SampleCount(test_pos, test_neg))

    return labeled_training_set, labeled_testing_set


def get_dataset_path(window_size, category, data_type, dataset=None):
    file_path = ''

    if category not in ['train', 'test']:
        raise ValueError('category must be either "train" or "test"')
    if data_type not in ['point', 'label']:
        raise ValueError('data_type must be either "point" or "label"')

    category_dir = f'/Users/adamsam/repos/ee/Pedestrian-Detection/datasets/npy_{category}'

    file_name = f'{data_type}_{window_size[1]}-{window_size[0]}.npy'

    if category == 'train':
        file_path = os.path.join(category_dir, file_name)
    elif category == 'test' and dataset is not None:
        file_path = os.path.join(category_dir, dataset, file_name)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    return file_path


def init_datasets(datasets_path):

    for window_size in window_sizes:
        total_training_points = np.array([])
        total_training_labels = np.array([])
        for dataset in ['INRIA', 'caltech_30', 'PnPLO']:
            print(f'\n\nInitializing dataset {dataset} with window size {window_size}\n')
            training_set, testing_set = prepare_labeled_datasets(os.path.join(datasets_path, dataset), window_size)

            print("Training Positives: ", training_set.sample_count.pos)
            print("Training Negatives: ", training_set.sample_count.neg)
            print("Testing Positives: ", testing_set.sample_count.pos)
            print("Testing Negatives: ", testing_set.sample_count.neg)


            if total_training_points.shape[0] == 0:
                total_training_points = training_set.points
                total_training_labels = training_set.labels
            else:
                total_training_points = np.concatenate((total_training_points, training_set.points))
                total_training_labels = np.concatenate((total_training_labels, training_set.labels))

            np.save(get_dataset_path(window_size, 'test', 'point', dataset), testing_set.points)
            np.save(get_dataset_path(window_size, 'test', 'label', dataset), testing_set.labels)

            print("\nInitialized")

        print("\n\nSaving total training sets\n")
        np.save(get_dataset_path(window_size, 'train', 'point'), total_training_points)
        np.save(get_dataset_path(window_size, 'train', 'label'), total_training_labels)


if __name__ == '__main__':
    init_datasets('../../datasets')



