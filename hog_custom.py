import cv2
import argparse
import os, shutil
import joblib
import json
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import SVC
from utils import *

def my_svm_detector(args):

    template_size = [128,64] # H, W  -> Standard Size as per paper
    step_size = (10,10) # For sliding window
    confidence_threshold = 0.75 # For Positive Prediction
    scale_factor = 1.25 # For Gaussian Pyramid
    
    if os.path.exists(str(args.inp_folder)+'/saved_models/trained_svm.pkl'):
        classifier =  joblib.load(str(args.inp_folder)+"/saved_models/trained_svm.pkl")
        print("Trained model found at saved_models/trained_svm.pk. Loading it...")
    else:
        print("Trained model NOT saved_models/trained_svm.pk.\nTraining New...")
        # Get positive & negative samples of given template size
        img,Y = prepare_data(args,template_size)
        X = []        
        # Calculating HoG Descriptors
        print("Computing HoG features")
        for i,x in  enumerate(tqdm(img)):
            x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
            
            # Compute HoG feature descriptor
            # fd, hog_image = hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2',visualize=True, feature_vector=True)
            fd = hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)

            """
            # Visualize HoG
            if Y[i]==1 and i%10==0:
                import matplotlib.pyplot as plt
                plt.imshow(hog_image,cmap='gray')
                plt.title("HOG")
                plt.show()
                cv2.waitKey(0)
            """

            X.append(fd)


        train_x = np.array(X)
        train_y = np.array(Y)
        # Support Vector Machine for Classification
        classifier = SVC (kernel='rbf',gamma='scale')
        print("Training SVM Classifier")
        classifier.fit(train_x,train_y)
        # Save Model
        if not os.path.exists(str(args.inp_folder)+'/saved_models'):
            os.mkdir(str(args.inp_folder)+'/saved_models')
        print(f"Saved Trained Model as {args.inp_folder}+'/saved_models/trained_svm.pkl'")
        joblib.dump( classifier, str(args.inp_folder)+"/saved_models/trained_svm.pkl")

    print("Testing the model...")
    images = sorted(os.listdir(os.path.join(args.inp_folder,"PNGImages")))
    if os.path.exists(str(args.inp_folder)+"/vis_b"):
        shutil.rmtree(str(args.inp_folder)+"/vis_b")
    os.mkdir(str(args.inp_folder)+"/vis_b")

    coco_result = {}
    coco_result['info'] = "Output Of Pedestrian Detection using our trained SVM"
    coco_result['images'] = []
    coco_result['detections'] = []
    image_id = 0
    category_id = 1
    

    for file in tqdm(images):
        rects = []
        confidence = []
        image_id+=1
        coco_result['images'].append({"file_name":file,"image_id":image_id})
        image = cv2.imread(os.path.join(args.inp_folder,"PNGImages",file))
        
        # Original Dimensions
        h,w = image.shape[0],image.shape[1]

        original_image = image.copy()

        # Resize for speed
        image = cv2.resize(image,(400,256))
        h_ratio = h/256
        w_ratio = w/400

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # List to store the detections
        rects = []
        confidence= []
        # For storing current scale
        scale = 0
        for im_scaled in pyramid_gaussian(image, scale_factor):
            #The list contains detections at the current scale
            if im_scaled.shape[0] > template_size[0] and im_scaled.shape[1] and template_size[1]:
                # Sliding Window for each level of pyramud
                windows = sliding_window(im_scaled, template_size, step_size)
                for (x, y, window) in windows:
                    if window.shape[0] == template_size[0] and window.shape[1] == template_size[1]:
                        fd=hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
                        fd = fd.reshape(1, -1)
                        pred = classifier.predict(fd)
                        if pred == 1:
                            confidence_score = classifier.decision_function(fd)
                            if confidence_score > confidence_threshold:
                                x1 = int(x * (scale_factor**scale)*w_ratio) # x_tl_in_original_image
                                y1 = int(y * (scale_factor**scale)*h_ratio) # y_tl_in_original_image
                                w_in_original_image = int(template_size[1] * (scale_factor**scale)*w_ratio)
                                h_in_original_image = int(template_size[0] * (scale_factor**scale)*h_ratio)
                                x3 = x1 + w_in_original_image # x_br_in_original_image
                                y3 = y1 + h_in_original_image # y_br_in_original_image
                                rects.append([x1,y1,x3,y3])
                                confidence.append([confidence_score])
            
                scale += 1
            else:
                break

        # Apply Non-max suppression
        rects,scores = NMS(rects,confidence)

        for rect,score in zip(rects,scores):
            x1,y1,x3,y3 = rect.tolist()
            coco_result['detections'].append({"image_id":image_id,"category_id":category_id,"bbox":[x1,y1,x3-x1,y3-y1],"score":score.item()})
            if args.vis:
                cv2.rectangle(original_image, (x1, y1), (x3, y3), (0, 0, 255), 2)
        if args.vis:
            cv2.imwrite(str(args.inp_folder)+"/vis_b/"+str(file),original_image)
        
    print(f"Saved predictions at {args.inp_folder+'/pred_hog_custom.json'}")
    json.dump(coco_result, open(args.inp_folder+"/pred_hog_custom.json", 'w'), ensure_ascii=False)


if __name__ == "__main__":
    argument_parser_object = argparse.ArgumentParser(description="Pedestrian Detection in images")
    argument_parser_object.add_argument('-i', '--inp_folder', type=str, default='PennFudanPed', help="Path for the root folder of dataset containing images, annotations etc.)")
    argument_parser_object.add_argument('-v', '--vis', action='store_true', default=False, help="Visualize Results (Add --vis to visualize")
    args = argument_parser_object.parse_args()
    my_svm_detector(args)