""" 
Useful Functions for COL780 Assignment 3
"""

import cv2
import os, shutil
import random
import json
import numpy as np
from tqdm import tqdm

from debug import get_single_debug_image_name

valid_objects = [
    'person',
    'people'
]

def get_pascal_voc_text_data(file_name):
    bbox = []
    with open(file_name) as f:
        for line in f:
            if "Bounding box" in line:
                line = line.split(":")[1].strip("\n").strip().replace(" ","")
                coordinate_arr = line.split("-")
                x1,y1 = map( lambda x:int(x) ,coordinate_arr[0].replace(")","").replace("(","").split(","))
                x3,y3 = map( lambda x:int(x) ,coordinate_arr[1].replace(")","").replace("(","").split(","))
                bbox.append([x1,y1,x3,y3])
    return bbox

def get_pascal_voc_xml_data(file_name):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_name)
    root = tree.getroot()
    bbox = []

    for obj in root.findall('object'):
        if not obj.find("name").text.lower() in valid_objects:
            continue
        bndbox = obj.find('bndbox')
        bbox.append([
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)
        ])
    return bbox


# Prepare Data For training SVM
def prepare_data(image_folder, window_size):

    image_dir = os.path.join(image_folder, "frame")
    annotation_dir = os.path.join(image_folder, "annotations")

    image_subdirs = [
        os.path.join(image_dir, subdir, "frame")
        for subdir in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, subdir))
    ]

    # images = [file for subdir in image_subdirs for file in os.listdir(subdir))]
    images = [os.path.join(subdir, file) for subdir in image_subdirs for file in os.listdir(subdir) if
              os.path.isfile(os.path.join(subdir, file))]

    # images = [file for file in os.listdir(image_dir) if not file.startswith(get_single_debug_image_name())]

    print("\nPreparing Training Data...")

    training_data_points = [] #trainx containes images
    training_labels = [] #contains labels, 1 for postive images and 0 for negative images
    num_pos=0
    num_neg=0

    for num,image_file_location in enumerate(tqdm(images)):
        print("Reading file name: ", image_file_location)
        image = cv2.imread(image_file_location)

        partial_location = image_file_location.split(os.sep)[-3:]
        partial_location[1] = 'bbox'
        annotation_file_location = os.path.join(
            annotation_dir,
            "/".join(map(str, partial_location))
        )[:-4]

        if os.path.exists(annotation_file_location+".txt"):
            bbox_arr = get_pascal_voc_text_data(annotation_file_location+".txt")
        elif os.path.exists(annotation_file_location+".xml"):
            bbox_arr = get_pascal_voc_xml_data(annotation_file_location+".xml")
        else:
            raise Exception("Annotation file not found")
        print("appending negative samples")
        for _ in range(3):
            h, w = image.shape[:2]
            if h > window_size[0] or w > window_size[1]:
                h = h - window_size[0];
                w = w - window_size[1]

                overlap = [True for i in bbox_arr]
                max_loop = 0
                while np.any(overlap) ==True:
                    max_loop+=1
                    if max_loop==10:
                        break
                    overlap = [True for i in bbox_arr]
                    x = random.randint(0, w)
                    y = random.randint(0, h)
                    window = [x, y, x + window_size[1], y + window_size[0]]
                    for var,bbox in enumerate(bbox_arr):
                        dx = min(bbox[2], window[2]) - max(bbox [0], window[0])
                        dy = min(bbox[3], window[3]) - max(bbox[1],  window[1])
                        if dx<=0 or dy<=0:
                            overlap[var] = False
                if max_loop<10:
                    img = image[window[1]:window[3],window[0]:window[2]]
                    training_data_points.append(img)
                    training_labels.append(0)
                    num_neg+=1
        print("appending positive samples")
        for box in bbox_arr:
            img = image[box[1]:box[3],box[0]:box[2]]
            training_data_points.append(img)
            training_labels.append(1)
            num_pos+=1
    print("resizing data points")
    training_data_points = [cv2.resize(data_point, (window_size[1], window_size[0])) for data_point in training_data_points]
    print(f"Prepared {num_pos} positive & {num_neg} negative training examples")

    # if args.vis:
    #     print(f"Saving training images in {str(image_folder)+'/train_data_hog_custom'}")
    #     if os.path.exists(str(image_folder)+"/train_data_hog_custom"):
    #         shutil.rmtree(str(image_folder)+"/train_data_hog_custom")
    #     os.mkdir(str(image_folder)+"/train_data_hog_custom")
    #     for num_sample,img in enumerate(train_x):
    #         cv2.imwrite(str(image_folder)+"/train_data_hog_custom/"+str(train_y[num_sample])+"_"+str(num_sample)+".png",img)

    return training_data_points,training_labels

def NMS(boxes, confidence,th = 0.3):
    if len(boxes) == 0:
        return np.array([], dtype=int),np.array([], dtype=float)
    rects_with_confidence = [[boxes[i],confidence[i]] for i in range(len(boxes))]

    # Sort according to confidence
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

# Sliding Window
def sliding_window(image, window_size, step_size):
    res_windows = []
    for y in range(0, image.shape[0], step_size[0]):
        for x in range(0, image.shape[1], step_size[1]):
            res_windows.append([x, y, image[y: y + window_size[0], x: x + window_size[1]]])
    return res_windows

def get_grayscale_image(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

import cv2
def get_grayscale_image(image):
   return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)