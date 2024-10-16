import os.path
import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import get_dataset_path
from hog import HOG_Parameters
from svm import SVM_Parameters, train_svm, load_svm
import matplotlib.pyplot as plt

from transform import grayscale_transform, hog_transform

def test_regular():
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
        step_size=(10, 10),
        hog_parameters=hog_parameters
    )

    svm = load_svm(
        svm_parameters,
        "../saved_models",
    )

    print("Loading test data")
    x_test = np.load(get_dataset_path((112,48), 'test', 'point', 'INRIA'))
    y_test = np.load(get_dataset_path((112,48), 'test', 'label', 'INRIA'))
    print("Transforming test data")
    x_test_gray = grayscale_transform(x_test)
    x_test_hog = hog_transform(x_test_gray, hog_parameters)
    print("Predicting")
    y_pred = svm.predict(x_test_hog)
    print(np.array(y_test == y_pred).mean())

def train_optimized():
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
        step_size=(10, 10),
        hog_parameters=hog_parameters
    )

    p_grid = {
        "C": [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
    }

    NUM_TRIALS  = 30

    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    X_train = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'point'))
    y_train = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'label'))
    print("Transforming train data points")
    X_train_gray = grayscale_transform(X_train)
    X_train_hog = hog_transform(X_train_gray, hog_parameters)

    svm = SVC(kernel='linear')

    for i in range(NUM_TRIALS):
        print(f"Trial {i+1}/{NUM_TRIALS}")
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=outer_cv)
        clf.fit(X_train_hog, y_train)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
        nested_score = cross_val_score(clf, X=X_train_hog, y=y_train, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    score_difference = non_nested_scores - nested_scores
    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    plt.subplot(211)
    (non_nested_scores_line,) = plt.plot(non_nested_scores, color="r")
    (nested_line,) = plt.plot(nested_scores, color="b")
    plt.ylabel("score", fontsize="14")
    plt.legend(
        [non_nested_scores_line, nested_line],
        ["Non-Nested CV", "Nested CV"],
        bbox_to_anchor=(0, 0.4, 0.5, 0),
    )
    plt.title(
        "Non-Nested and Nested Cross Validation on Iris Dataset",
        x=0.5,
        y=1.1,
        fontsize="15",
    )

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend(
        [difference_plot],
        ["Non-Nested CV - Nested CV Score"],
        bbox_to_anchor=(0, 1, 0.8, 0),
    )
    plt.ylabel("score difference", fontsize="14")

    plt.show()

def train_compare():
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
        step_size=(10, 10),
        hog_parameters=hog_parameters
    )
    X_raw = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'point'))
    y_train = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'label'))
    X_train = hog_transform(grayscale_transform(X_raw), hog_parameters)

    svm_hard = SVC(kernel='linear', C=1)
    svm_soft = SVC(kernel='linear', C=0.01)

    print("Training hard SVM")
    svm_hard.fit(X_train, y_train)
    print("Training soft SVM")
    svm_soft.fit(X_train, y_train)

    for dataset in ['INRIA', 'caltech_30', 'PnPLO']:
        print(f"Loading {dataset}")
        X_raw = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset))
        y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', dataset))
        X_test = hog_transform(grayscale_transform(X_raw), hog_parameters)

        print(f"Getting model predictions for {dataset}")
        y_pred_hard = svm_hard.predict(X_test)
        y_pred_soft = svm_soft.predict(X_test)

        print("Hard SVM Accuracy: ", np.array(y_test == y_pred_hard).mean())
        print("Soft SVM Accuracy: ", np.array(y_test == y_pred_soft).mean())

def train_sgd():
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
        step_size=(10, 10),
        hog_parameters=hog_parameters
    )


    X_raw = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'point'))
    y_train = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'label'))
    X_train = hog_transform(grayscale_transform(X_raw), hog_parameters)

    X_raw_inria = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', 'INRIA'))
    y_test_inria = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', 'INRIA'))
    X_test_inria = hog_transform(grayscale_transform(X_raw_inria), hog_parameters)


    x_raw_caltech = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', 'caltech_30'))
    y_test_caltech = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', 'caltech_30'))
    x_test_caltech = hog_transform(grayscale_transform(x_raw_caltech), hog_parameters)

    x_raw_pnplo = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', 'PnPLO'))
    y_test_pnplo = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', 'PnPLO'))
    x_test_pnplo = hog_transform(grayscale_transform(x_raw_pnplo), hog_parameters)

    scalify = StandardScaler()
    X_train_scaled = scalify.fit_transform(X_train)
    sgd_standard = SGDClassifier()
    sgd_standard.fit(X_train_scaled, y_train)

    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)

    for X, y in [(x_test_caltech, y_test_caltech), (x_test_pnplo, y_test_pnplo), (X_test_inria, y_test_inria)]:
        y_pred_standard = sgd_standard.predict(scalify.transform(X))
        y_pred = sgd.predict(X)
        print("Standard Accuracy: ", np.array(y == y_pred_standard).mean())
        print("SGD Accuracy: ", np.array(y == y_pred).mean())
        print('-------------------')
    # print("Loading test data")
    # x_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', 'INRIA'))
    # y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', 'INRIA'))

    # print("Transforming test data")
    # x_test_gray = grayscale_transform(x_test)
    # x_test_hog = hog_transform(x_test_gray, hog_parameters)
    # print("Predicting")
    # y_pred = svm.predict(x_test_hog)
    # print(np.array(y_test == y_pred).mean())
# train_sgd
window_sizes = [(100, 50), (128, 96), (128, 64), (112, 48)]
orientations = [9, 13, 18]
# (4,4), (6,6), (8,8)
pixels_per_cell_list = [(10,10)]
cells_per_block_list = [(1,1),(2,2), (3,3), (4,4)]
block_strides = [(1,1), (2,2), (3,3)]
holistic_derivative_masks = [True, False]

def get_model_count():
    count = 0
    for _ in window_sizes:
        for _ in orientations:
            for _ in pixels_per_cell_list:
                for cells_per_block in cells_per_block_list:
                    for block_stride in block_strides:
                        if block_stride[0] > cells_per_block[0]:
                            continue
                        for _ in holistic_derivative_masks:
                            count += 1
    return count

def prep_models():
    total_model_count = len(window_sizes) * len(orientations) * len(pixels_per_cell_list) * len(
        cells_per_block_list) * len(block_strides) * len(holistic_derivative_masks)
    current_iteration = 0

    for window_size in window_sizes:
        for orientation_bins in orientations:
            for pixels_per_cell in pixels_per_cell_list:
                for cells_per_block in cells_per_block_list:
                    for block_stride in block_strides:
                        if block_stride[0] > cells_per_block[0]:
                            print("\nBlock stride cannot be greater than cells per block\n")
                            continue
                        for holistic_derivative_mask in holistic_derivative_masks:
                            hog_parameters = HOG_Parameters(
                                orientations=orientation_bins,
                                pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block,
                                block_stride=block_stride,
                                holistic_derivative_mask=holistic_derivative_mask,
                                block_norm='L2-Hys'
                            )
                            svm_parameters = SVM_Parameters(
                                window_size=window_size,
                                hog_parameters=hog_parameters
                            )
                            current_iteration += 1
                            print(f"Training ${current_iteration}/{total_model_count}")
                            print(svm_parameters.get_svm_name())
                            train_svm(svm_parameters,
                                      get_dataset_path(svm_parameters.window_size, 'train', 'point'),
                                      get_dataset_path(svm_parameters.window_size, 'train', 'label'))

if __name__ == "__main__":
    prep_models()