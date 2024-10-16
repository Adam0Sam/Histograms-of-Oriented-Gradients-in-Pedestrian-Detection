
import cv2
import os, shutil
import random
import json
import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

valid_objects = [
    'person',
    'people'
]

def train():
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(hog_parameters)
    scalify = StandardScaler()

    X_train, y_train = prepare_labeled_data('../datasets/caltech_30/Train', window_size)

    # X_train = np.load('../datasets/playground/train/train_x.npy')
    # y_train = np.load('../datasets/playground/train/train_y.npy')

    print("Grayfiying")
    X_train_gray = grayify.fit_transform(X_train)
    print("Hogifying")
    X_train_hog = hogify.fit_transform(X_train_gray)
    print("Scalifying")
    # X_train_prepared = scalify.fit_transform(X_train_hog)

    print("Training")
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_hog, y_train)

    print("Testing")

    # X_test = np.load('../datasets/playground/test/test_x.npy')
    # y_test = np.load('../datasets/playground/test/test_y.npy')

    print("Preparing test from caltech")

    X_test, y_test = prepare_labeled_data('../datasets/caltech_30/Test', window_size)

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    # X_test_prepared = scalify.transform(X_test_hog)

    y_pred = sgd_clf.predict(X_test_hog)
    print(np.array(y_test == y_pred).mean())

def NMS(boxes, confidence,th = 0.3):
    if len(boxes) == 0:
        return np.array([], dtype=int),np.array([], dtype=float)
    rects_with_confidence = [[boxes[i],confidence[i]] for i in range(len(boxes))]

    rects_with_confidence = (sorted(rects_with_confidence, key=lambda box: box[1][0],reverse=True))

    rects = [var[0] for var in rects_with_confidence]
    
    bool_arr = [True for i in rects_with_confidence]
    
    for i,box in enumerate(rects):
        if bool_arr[i] == True:
            for j,other_box in enumerate(rects[i+1:]):
                k = j+i+1
                if bool_arr[k] == True:
                    dx = max(0,min(box[2], other_box[2]) - max(box [0], other_box[0]))
                    dy = max(0,min(box[3], other_box[3]) - max(box[1], other_box[1]))
                    
                    overlap = float(dx*dy)
                    overlap_percentage = overlap/((other_box[3]-other_box[1])*(other_box[2]-other_box[0]))
                    if overlap_percentage > th:
                        bool_arr[k] = False
                    
    
    final_rects = []
    final_confidence = []
    for i,rect in enumerate(rects):
        if bool_arr[i]:
            final_rects.append(rect)
            final_confidence.append(rects_with_confidence[i][1][0])
    
    return np.array(final_rects, dtype=int),np.array(final_confidence, dtype=float)

def sliding_window(image, window_size, step_size):
    res_windows = []
    for y in range(0, image.shape[0], step_size[0]):
        for x in range(0, image.shape[1], step_size[1]):
            res_windows.append([x, y, image[y: y + window_size[0], x: x + window_size[1]]])
    return res_windows

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

    for frame_subdir in tqdm(os.listdir(frame_dir)):
        frame_subdir_path = os.path.join(frame_dir, frame_subdir)
        if(os.path.isdir(frame_subdir_path)):
            frame_files = os.listdir(os.path.join(frame_subdir_path, 'frame'))
            for frame_file in frame_files:
                file_location = os.path.join(frame_subdir_path, 'frame', frame_file)

                if not os.path.isfile(file_location):
                    continue

                frame_number = int(frame_file.split('.')[0].split('_')[-1])

                if frame_number % 30 != 0:
                    os.remove(file_location)
                    annotation_file_location = os.path.join(annotation_dir, frame_subdir, 'bbox', frame_file.split('.')[0] + '.xml')
                    if os.path.isfile(annotation_file_location):
                        os.remove(annotation_file_location)


if __name__ == "__main__":
    caltech_data_points, caltech_labels = prepare_data("../datasets/caltech_30/Train", (128, 64))
    inria_data_points, inria_labels = prepare_data("../datasets/INRIA/Train", (128, 64))
    pnplo_data_points, pnplo_labels = prepare_data("../datasets/PnPLO/Train", (128, 64))

    np.save(
        "../datasets/data_points.npy",
        np.array(caltech_data_points + inria_data_points + pnplo_data_points)
    )
    np.save("../datasets/labels.npy", np.array(caltech_labels + inria_labels + pnplo_labels))

