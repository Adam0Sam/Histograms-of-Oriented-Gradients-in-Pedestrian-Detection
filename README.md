# Pedestrian-Detection
Given an input image which may contain multiple pedestrians, a pedestrian detection system detects each person and returns a bounding box for each detection

![image](https://user-images.githubusercontent.com/66300465/145832492-76d97fe6-bd2d-4b57-94f7-e5c40dce35ca.png)

This repo uses OpenCV to do template tracking using 2 methods:
* ***HoG Method:***
  - Firstly, we use a pretrained HoG detector (trained on the INRIA dataset as per the original paper [Histograms of Oriented Gradients for Human Detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467360)). For this, we use OpenCV’s [HOGDescriptor()](https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html) with [getDefaultPeopleDetector()](https://docs.opencv.org/4.5.3/d5/d33/structcv_1_1HOGDescriptor.html#a9c7a0b2aa72cf39b4b32b3eddea78203)

  - Next, we extract HoG features using [skimage's HoG feature extractor](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html) and train our own SVM classifier on top of it. For this, prepare the training data with positive and negative samples for SVM.

* ***Faster-RCNN Method:*** Here, we use the pre-trained Faster-RCNN detector (as described in [Faster r-cnn: Towards real-time object detection with region proposal networks](https://arxiv.org/pdf/1506.01497.pdf)).
  - We implenmted PyTorch's implementation of Faster-RCNN. Here's a link to [PyTorch Faster-RCNN Tutorial](https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/)
  - The pre-trained model detects multiple object categories, but we are only concerned with the ‘person’ category. Since we used a model trained on the COCO dataset, we only needed the predictions for class label 1, which corresponds to the ‘person’ category

### Dataset ###
The dataset used is "PennFudanPed" and can be downloaded from [here](https://www.cis.upenn.edu/~jshi/ped_html) and can be stored in PennFudanPed folder

### Using This Repo  ###
* Download PennFudanPed dataset and put in [PennFudanPed](https://github.com/abdur75648/Pedestrian-Detection/tree/main/PennFudanPed) folder
* Install required python packages using: ` pip install -r requirements.txt `

#### Detection ####
* Run any of the three scripts using the corresponding command:

    - [hog_pretrained.py](https://github.com/abdur75648/Pedestrian-Detection/blob/main/hog_pretrained.py) : ` python hog_pretrained.py -i PennFudanPed --vis `,

    - [hog_custom.py](https://github.com/abdur75648/Pedestrian-Detection/blob/main/hog_custom.py) : ` python hog_custom.py -i PennFudanPed --vis `,

    - [faster_rcnn.py](https://github.com/abdur75648/Pedestrian-Detection/blob/main/faster_rcnn.py) : ` python faster_rcnn.py -i PennFudanPed --vis `,
  
  where -i is dataset folder path. Add --vis to visualize detections
  
#### Evalutaion ####
* The following metrics have been provided for evaluation:
  - Average Precision (AP): AP evaluated and averaged over 10 IoU thresholds of .50:.05:.95
  - Average Recall (AR): AR averaged over IoUs and evaluated at 1 and 10 detections per image.

* Results can be evaluated using [evaluation script](https://github.com/abdur75648/Pedestrian-Detection/blob/main/evaluate_detections.py) provided. Use the command:
 
    ` python evaluate_detections.py --gt <path to ground truth annotations json> --pred <path to detections json> `
