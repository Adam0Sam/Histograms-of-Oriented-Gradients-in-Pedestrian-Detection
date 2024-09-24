import os.path
import cv2
from hog import HOG_Parameters
from svm import SVM_Parameters, train_svm, get_model_predictions
from utils import prepare_data

window_sizes = [(128, 64), (96, 48), (160, 80), (192, 96), (112,48), (144, 72), (88,176), (64,128), (48,96), (32,64)]
step_sizes = [(10, 10), (8, 8), (12, 12), (16, 16), (8, 16), (16, 8), (8, 4), (4, 8), (4, 4), (2, 2)]
orientations = [9, 7, 11, 13, 15, 17, 18, 19, 20, 21, 22]
pixels_per_cell = [(8, 8), (6, 6), (10, 10), (12, 12), (14, 14), (16, 16), (2,1), (1,2), (4,4), (2,2)]
cells_per_block = [(2, 2), (1, 1), (3, 3), (4, 4), (5, 5), (6, 6), (1,2), (2,1), (1,1), (2,2)]
block_strides = [(1, 1), (2, 2), (3, 3), (4, 4), (1,2), (2,1), (1,1), (2,2)]

holistic_derivative_masks = [True, False]
block_norms = ['L1', 'L1-sqrt', 'L2', 'L2-Hys']


def main(dataset_dir):
    hog_parameters = HOG_Parameters(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_stride=(1, 1),
        holistic_derivative_mask=False,
        block_norm='L2-Hys'
    )
    svm_parameters = SVM_Parameters(
        window_size=(128, 64),
        step_size=(10,10),
        hog_parameters=hog_parameters
    )
    print("STAGE 1: Fetching SVM")
    train_svm(svm_parameters, dataset_dir, force_train=True, save_model=True)
    print("STAGE 2: Getting Model Predictions")
    results = get_model_predictions(svm_parameters, dataset_dir)
    print(results)

    for test_image, detection in zip(results['test_images'], results['detections']):
        image = cv2.imread(os.path.join(dataset_dir, 'PNGImages', test_image['file_name']))
        image = image.copy()
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Detection', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def prepare_all_models():
    for window_size in window_sizes:
        for step_size in step_sizes:
            for orientation in orientations:
                for pixel_per_cell in pixels_per_cell:
                    for cell_per_block in cells_per_block:
                        for block_stride in block_strides:
                            for holistic_derivative_mask in holistic_derivative_masks:
                                for block_norm in block_norms:
                                    hog_parameters = HOG_Parameters(
                                        orientations=orientation,
                                        pixels_per_cell=pixel_per_cell,
                                        cells_per_block=cell_per_block,
                                        block_stride=block_stride,
                                        holistic_derivative_mask=holistic_derivative_mask,
                                        block_norm=block_norm
                                    )
                                    for dataset_dir in ["datasets/Penn-Fudan", "datasets/INRIA"]:
                                        svm_parameters = SVM_Parameters(
                                            window_size=window_size,
                                            step_size=step_size,
                                            hog_parameters=hog_parameters
                                        )
                                        print("STAGE 1: Fetching SVM")
                                        train_svm(svm_parameters, dataset_dir, force_train=True, save_model=True)
                                        print("STAGE 2: Getting Model Predictions")
                                        results = get_model_predictions(svm_parameters, dataset_dir)
                                        print(results)

                                        for test_image, detection in zip(results['test_images'], results['detections']):
                                            image = cv2.imread(os.path.join(dataset_dir, 'PNGImages', test_image['file_name']))
                                            image = image.copy()
                                            x1, y1, x2, y2 = detection['bbox']
                                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                            cv2.imshow('Detection', image)
                                            cv2.waitKey(0)
                                        cv2.destroyAllWindows()


def test():
    hog_parameters = HOG_Parameters(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_stride=(1, 1),
        holistic_derivative_mask=False,
        block_norm='L2-Hys'
    )
    svm_parameters = SVM_Parameters(
        window_size=(128, 64),
        step_size=(10,10),
        hog_parameters=hog_parameters
    )
    train_svm(svm_parameters, "../datasets/caltech_parsed/Train", force_train=True, save_model=True)

def test2():
    prepare_data("../datasets/caltech_parsed/Train", (128, 64))
if __name__ == "__main__":
    # prepare_all_models()
    # test()
    test2()
    # dataset_dir = "datasets/Penn-Fudan"
    # main(dataset_dir)