import os.path
import cv2
from hog import HOG_Parameters
from svm import SVM_Parameters, train_svm, get_model_predictions
from utils import prepare_data

window_sizes = [(128, 64), (96, 48), (112,48), (144, 72), (196, 128)]
step_sizes = [(10, 10), (8, 8), (12, 12), (16, 16), (8, 16), (4, 4), (2, 2)]
orientations = [9, 13, 18]
pixels_per_cell = [(8, 8), (6, 6), (10, 10), (12, 12), (14, 14), (16, 16), (4,4), (2,2), (1,1)]
cells_per_block = [(2, 2), (1, 1), (3, 3), (4, 4), (5, 5), (6, 6)]
block_strides = [(1, 1), (2, 2), (3, 3), (4, 4)]
holistic_derivative_masks = [True, False]


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


def count():
    print("Window Sizes: ", len(window_sizes))
    print("Step Sizes: ", len(step_sizes))
    print("Orientations: ", len(orientations))
    print("Pixels Per Cell: ", len(pixels_per_cell))
    print("Cells Per Block: ", len(cells_per_block))
    print("Block Strides: ", len(block_strides))
    print("Holistic Derivative Masks: ", len(holistic_derivative_masks))
    print("------------------- Total -------------------")
    largest_window_dimensions = max(window_sizes, key=lambda x: x[0] * x[1])
    smallest_block_dimensions = min(cells_per_block, key=lambda x: x[0] * x[1])
    biggest_block_strides = max(block_strides, key=lambda x: x[0] * x[1])
    print("Total Models: ", len(window_sizes) * len(orientations) * len(pixels_per_cell) * len(cells_per_block) * len(block_strides) * len(holistic_derivative_masks))
    print("Largest Feature Vector: ", ((largest_window_dimensions[0]-smallest_block_dimensions[0]*biggest_block_strides[0])*(largest_window_dimensions[1]-smallest_block_dimensions[1]*biggest_block_strides[1])*9)/(biggest_block_strides[0]*biggest_block_strides[1]))

if __name__ == "__main__":
    # prepare_all_models()
    # test()
    count()
    # dataset_dir = "datasets/Penn-Fudan"
    # main(dataset_dir)