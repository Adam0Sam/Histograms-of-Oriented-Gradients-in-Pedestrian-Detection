
def old_get_model_predictions(svm_parameters: SVM_Parameters, image_folder):
    classifier = load_svm(svm_parameters, "../saved_models")

    test_image_names = ['red.webp']

    SCALE_FACTOR = 1.25
    CONFIDENCE_THRESHOLD = 1
    results = {'test_images': [], 'detections': []}

    image_id = 0
    category_id = 1

    for image_name in tqdm(test_image_names):

        image_id += 1
        results['test_images'].append({
            "file_name": image_name,
            "image_id": image_id
        })

        image = cv2.imread(os.path.join(image_folder, image_name))

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
                    if prediction[0] == 1:
                        confidence_score = classifier.decision_function(feature_vector)

                        print(f"Confidence: {confidence_score[0]}")
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

        # rects,scores = NMS(rects,confidence)
        # for rect,score in zip(rects,scores):
        #     x1,y1,x2,y2 = rect.tolist()
        #     results['detections'].append({"image_id":image_id,"category_id":category_id,"bbox":[x1,y1,x2-x1,y2-y1],"score":score.item()})

    return results


def see_dimensionality():
    for pixels_per_cell in [(4, 4), (6, 6), (8, 8), (10, 10)]:
        hog_parameters = HOG_Parameters(
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            block_stride=(1, 1),
            holistic_derivative_mask=False,
            block_norm='L2-Hys'
        )
        X_raw = np.load(get_dataset_path((128, 64), 'train', 'point'))
        X_train = hog_transform(grayscale_transform(X_raw), hog_parameters)
        print(f"Pixels per cell: {pixels_per_cell}")
        print(f"Dimensionality: {X_train.shape}")

def see_dimensionality_orientations():
    for orientations in [9, 13, 18]:
        hog_parameters = HOG_Parameters(
            orientations=orientations,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_stride=(1, 1),
            holistic_derivative_mask=False,
            block_norm='L2-Hys'
        )
        X_raw = np.load(get_dataset_path((128, 64), 'train', 'point'))
        X_train = hog_transform(grayscale_transform(X_raw), hog_parameters)
        print(f"Orientations: {orientations}")
        print(f"Dimensionality: {X_train.shape}")

def see_dimensionality_block_sizes():
    for cells_per_block in [(3, 3), (4, 4)]:
        hog_parameters = HOG_Parameters(
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=cells_per_block,
            block_stride=(1, 1),
            holistic_derivative_mask=False,
            block_norm='L2-Hys'
        )
        X_raw = np.load(get_dataset_path((128, 64), 'train', 'point'))
        X_train = hog_transform(grayscale_transform(X_raw), hog_parameters)
        print(f"Cells per block: {cells_per_block}")
        print(f"Dimensionality: {X_train.shape}")

def see_dimensionality_block_strides():
    for block_stride in [(1, 1), (2, 2), (3, 3)]:
        hog_parameters = HOG_Parameters(
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_stride=block_stride,
            holistic_derivative_mask=False,
            block_norm='L2-Hys'
        )
        print(f"\nBlock stride: {block_stride}")
        X_raw = np.load(get_dataset_path((128, 64), 'train', 'point'))
        print("X_raw shape: ", hog(grayscale_transform(X_raw)[0], hog_parameters).shape)
        # X_train = hog_transform(grayscale_transform(X_raw[0]), hog_parameters)
        # print(f"Dimensionality: {X_train.shape}")


def eval():
    window_sizes = [(100, 50), (128, 64)]
    block_strides = [(1, 1), (2, 2), (3,3)]
    for window_size in window_sizes:
        for block_stride in block_strides:
            hog_parameters = HOG_Parameters(
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(4, 4),
                block_stride=block_stride,
                holistic_derivative_mask=False,
                block_norm='L2-Hys'
            )
            svm_parameters = SVM_Parameters(
                window_size=window_size,
                hog_parameters=hog_parameters
            )
            try:
                svm = load_svm(svm_parameters, "../saved_models")
            except:
                print(f"Model not found for {svm_parameters.get_svm_name()}")
                continue
            X_path = get_dataset_path(svm_parameters.window_size, 'test', 'point', 'INRIA')
            y_path = get_dataset_path(svm_parameters.window_size, 'test', 'label', 'INRIA')
            X_raw = np.load(X_path)
            X_test = hog_transform(grayscale_transform(X_raw), hog_parameters)
            y_test = np.load(y_path)
            print(svm_parameters.get_svm_name())
            print(svm.score(X_test, y_test))

def count_zero_arrays(data):
    zero_arrays_count = 0
    for array in data:
        if np.all(array == 0):
            zero_arrays_count += 1
    return zero_arrays_count

def idk():
    block_strides = [(2,2), (1,1)]
    for block_stride in block_strides:
        hog_parameters = HOG_Parameters(
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_stride=block_stride,
            holistic_derivative_mask=False,
            block_norm='L2-Hys'
        )
        svm_parameters = SVM_Parameters(
            window_size=(128, 64),
            hog_parameters=hog_parameters
        )
        print("Block stride: ", block_stride)

        X_train_raw = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'point'))
        y_train = np.load(get_dataset_path(svm_parameters.window_size, 'train', 'label'))
        X_train_m = m_hog_transform(grayscale_transform(X_train_raw), hog_parameters)

        svm = load_svm(svm_parameters, "../saved_models")
        svm_m = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        print("Training SGD with magnitude")
        svm_m.fit(X_train_m, y_train)

        for dataset in ['INRIA', 'caltech_30', 'PnPLO']:
            X_test_raw = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset))
            y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', dataset))
            X_test_m = m_hog_transform(grayscale_transform(X_test_raw), hog_parameters)
            X_test = hog_transform(grayscale_transform(X_test_raw), hog_parameters)
            print(dataset)
            print(
                "SGD_m: ",
                np.array(y_test == svm_m.predict(X_test_m)).mean()
            )

            print(
                "SGD: ",
                np.array(y_test == svm.predict(X_test)).mean()
            )
            print('\n')
        print('-------------------\n')

def exclusive_caltech_train():
    block_strides = [(1, 1), (2,2)]
    for block_stride in block_strides:
        hog_parameters = HOG_Parameters(
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_stride=block_stride,
            holistic_derivative_mask=False,
            block_norm='L2-Hys'
        )
        svm_parameters = SVM_Parameters(
            window_size=(128, 64),
            hog_parameters=hog_parameters
        )
        training_set, testing_set = prepare_labeled_datasets(os.path.join('../datasets', 'caltech_30'), svm_parameters.window_size)
        X_train = hog_transform(grayscale_transform(training_set.points), hog_parameters)
        y_train = training_set.labels

        X_test_1 = hog_transform(grayscale_transform(testing_set.points), hog_parameters)
        y_test_1 = testing_set.labels
        loaded_svm = load_svm(svm_parameters, "../saved_models")

        X_test_2 = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', 'caltech_30'))
        X_test_2 = hog_transform(grayscale_transform(X_test_2), hog_parameters)
        y_test_2 = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', 'caltech_30'))

        trained_svm = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        print("Training SGD")
        trained_svm.fit(X_train, y_train)

        print("Exclusively Trained SGD: ", np.array(y_test_1 == trained_svm.predict(X_test_1)).mean())
        print("Loaded SGD: ", np.array(y_test_2 == loaded_svm.predict(X_test_2)).mean())

def train_sgd_and_qp(window_sizes, cells_per_block_list):
    for window_size in window_sizes:
        for cells_per_block in cells_per_block_list:
            hog_parameters = HOG_Parameters(
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=cells_per_block,
                block_stride=(1, 1),
                holistic_derivative_mask=False,
                block_norm='L2-Hys'
            )
            svm_parameters = svm_parameters(
                window_size=window_size,
                hog_parameters=hog_parameters
            )

            X_path = get_dataset_path(svm_parameters.window_size, 'train', 'point')
            y_path = get_dataset_path(svm_parameters.window_size, 'train', 'label')

            X_raw = np.load(X_path)
            X_train = hog_transform(grayscale_transform(X_raw), hog_parameters)
            y_train = np.load(y_path)
            print(svm_parameters.get_svm_name())
            print("Training SVM with QP")
            svm_qp = SVC(kernel='linear', C=0.01)
            svm_qp.fit(X_train, y_train)
            joblib.dump(svm_qp, f"../saved_models/{svm_parameters.get_svm_name()}_qp.pkl")

            print("Training SVM with SGD")
            train_svm(svm_parameters, X_path, y_path)

def compare_sgd_to_qp():
    window_sizes = [(100, 50), (128, 64)]
    cells_per_block_list = [(2, 2), (3, 3)]
    for window_size in window_sizes:
        for cells_per_block in cells_per_block_list:
            hog_parameters = HOG_Parameters(
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=cells_per_block,
                block_stride=(1, 1),
                holistic_derivative_mask=False,
                block_norm='L2-Hys'
            )
            svm_parameters = svm_parameters(
                window_size=window_size,
                hog_parameters=hog_parameters
            )

            svm_qp = load_svm(svm_parameters, "../saved_models", f"{svm_parameters.get_svm_name()}_qp")
            svm_sgd = load_svm(svm_parameters, "../saved_models")

            X_raw_inria = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', 'INRIA'))
            y_test_inria = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', 'INRIA'))
            X_test_inria = hog_transform(grayscale_transform(X_raw_inria), hog_parameters)

            print(f"Comparing {svm_parameters.get_svm_name()}")
            y_dec_qp = svm_qp.predict(X_test_inria)
            y_dec_sgd = svm_sgd.predict(X_test_inria)
            print("QP Accuracy: ", np.array(y_test_inria == y_dec_qp).mean())
            print("SGD Accuracy: ", np.array(y_test_inria == y_dec_sgd).mean())

def compare_training_time():
    hog_parameters = HOG_Parameters(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_stride=(1, 1),
        holistic_derivative_mask=False,
        block_norm='L2-Hys'
    )

    X_train = hog_transform(grayscale_transform(np.load(get_dataset_path(
        (128, 64),
        'train',
        'point'
    ))), hog_parameters)
    y_train = np.load(get_dataset_path(
        (128, 64),
        'train',
        'label'
    ))

    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3)
    sgd_calibrated = CalibratedClassifierCV(sgd, cv=5)

    print("Training SGD")
    sgd_time_start = time.time()
    sgd.fit(X_train, y_train)
    print("Training SGD Time: ", time.time() - sgd_time_start)
    print("Training Calibrated SGD")
    sgd_calibrated_time_start = time.time()
    sgd_calibrated.fit(X_train, y_train)
    print("Training Calibrated SGD Time: ", time.time() - sgd_calibrated_time_start)


    for dataset in ['INRIA', 'caltech_30', 'PnPLO']:
        X_test = hog_transform(grayscale_transform(np.load(get_dataset_path(
            (128, 64),
            'test',
            'point',
            dataset
        ))), hog_parameters)
        y_test = np.load(get_dataset_path(
            (128, 64),
            'test',
            'label',
            dataset
        ))

        print("SGD Accuracy: ", sgd.score(X_test, y_test))
        print("Calibrated SGD Accuracy: ", sgd_calibrated.score(X_test, y_test))


def prep_models(cheap=True):

    total_model_count = get_model_count(
        cheap=cheap
    )
    current_iteration = 0

    def train_sgd(svm_parameters):
        hog_parameters = svm_parameters.hog_parameters
        if hog_parameters.block_stride[0] > hog_parameters.cells_per_block[0]:
            print("\nBlock stride cannot be greater than cells per block\n")
            return
        if cheap and hog_parameters.cells_per_block[0] >= hog_parameters.block_stride[0] * 2:
            print("Expensive Computation. Skipping...\n")
            return
        if not cheap and hog_parameters.cells_per_block[0] < hog_parameters.block_stride[0] * 2:
            print("Cheap Computation. Skipping...\n")
            return
        nonlocal current_iteration
        nonlocal total_model_count
        current_iteration += 1
        print(f"Training ${current_iteration}/{total_model_count}")
        print(svm_parameters.get_svm_name())
        train_svm(svm_parameters,
                  get_dataset_path(svm_parameters.window_size, 'train', 'point'),
                  get_dataset_path(svm_parameters.window_size, 'train', 'label'))

    iterate_model_parameters(train_sgd, cheap=cheap)

def prep_models_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cheap", action="store_true")
    parser.add_argument("--expensive", dest='cheap', action="store_false")
    args = parser.parse_args()
    prep_models(args.cheap)

def see_training_point_count():
    for window_size in window_sizes:
        X_path = get_dataset_path(window_size, 'train', 'point')
        X_raw = np.load(X_path)
        print(f"Window Size: {window_size}")
        print(f"Point Count: {X_raw.shape}\n")

def see_testing_point_count():
    for window_size in window_sizes:
        for dataset in ['INRIA', 'caltech_30', 'PnPLO']:
            X_path = get_dataset_path(window_size, 'test', 'point', dataset)
            X_raw = np.load(X_path)
            print(f"Window Size: {window_size}")
            print(f"Dataset: {dataset}")
            print(f"Point Count: {X_raw.shape}\n")


def see_dataset_balance():
    for window in window_sizes:
        y = np.load(get_dataset_path(
            window,
            'train',
            'label',
        ))
        print(f"Window: {window}")
        print(f"Positive: {np.sum(y)}")
        print(f"Negative: {y.shape[0] - np.sum(y)}\n")





def hog_vs_m_hog():
    hog_parameters = HOG_Parameters(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(4, 4),
        block_stride=(1, 1),
        holistic_derivative_mask=True,
        block_norm='L2-Hys'
    )
    svm_parameters = SVM_Parameters(
        window_size=(128, 64),
        hog_parameters=hog_parameters
    )
    sgd = load_svm(
        svm_parameters,
        '../computed/models'
    )
    sgd_m = SGDClassifier(loss='hinge')
    X_raw_train = np.load(get_dataset_path((128, 64), 'train', 'point'))
    X_gray_train = grayscale_transform(X_raw_train)
    X_train_m = m_hog_transform(X_gray_train, hog_parameters)
    y_train = np.load(get_dataset_path((128, 64), 'train', 'label'))

    sgd_m.fit(X_train_m, y_train)

    for dataset in datasets:
        X_raw_test = np.load(get_dataset_path((128, 64), 'test', 'point', dataset))
        X_gray_test = grayscale_transform(X_raw_test)
        X_test = hog_transform(X_gray_test, hog_parameters)
        X_test_m = m_hog_transform(X_gray_test, hog_parameters)
        y_test = np.load(get_dataset_path((128, 64), 'test', 'label', dataset))

        y_pred = sgd.predict(X_test)
        y_pred_m = sgd_m.predict(X_test_m)

        print(f"Dataset: {dataset}")
        print(f"SGD Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"SGD-M Accuracy: {accuracy_score(y_test, y_pred_m)}")
        print("\n")


import time

from dataset import get_dataset_path
from evaluate import evaluate_pedestrian_classifier
from parameters import HOG_Parameters, SVM_Parameters
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, loguniform
import joblib
import os
from datetime import datetime

from transform import hog_transform, grayscale_transform


def get_reg_values(min_exp=-4, max_exp=4, num_values=10):
    """
    Generate evenly spaced regularization values on a logarithmic scale.

    Parameters:
    -----------
    min_exp : int
        Minimum exponent for 10^min_exp (default: -4)
    max_exp : int
        Maximum exponent for 10^max_exp (default: 4)
    num_values : int
        Number of values to generate (default: 10)

    Returns:
    --------
    numpy.ndarray
        Array of regularization values
    """
    # Generate evenly spaced exponents
    exponents = np.linspace(min_exp, max_exp, num_values)

    # Convert to actual values using 10^exponent
    reg_values = 10 ** exponents

    return reg_values


hog_parameters = HOG_Parameters(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_stride=(1, 1),
    block_norm='L2-Hys',
    holistic_derivative_mask=False,
)

svm_parameters = SVM_Parameters(
    hog_parameters=hog_parameters,
    window_size=(128, 64)
)

X_raw = np.load(
    get_dataset_path(
        svm_parameters.window_size,
        'train',
        'point',
    )
)
y_train = np.load(
    get_dataset_path(
        svm_parameters.window_size,
        'train',
        'label',
    )
)
X_train = hog_transform(grayscale_transform(X_raw), hog_parameters)


def save_model(model, params, model_type, base_path='test'):
    """
    Save trained model and associated objects
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Save model and scaler
    model_path = os.path.join(base_path, f'{model_type}_model.pkl')
    params_path = os.path.join(base_path, f'{model_type}_params.pkl')

    joblib.dump(model, model_path)
    joblib.dump(params, params_path)

    return model_path, params_path


def train_libsvm(svm_parameters, n_iter=50):
    """
    Train SVM using libsvm (SVC) with hyperparameter optimization
    """
    print("Training LibSVM model...")

    svm = SVC(random_state=42, C=0.01)
    svm.fit(X_train, y_train)
    evaluate_pedestrian_classifier(
        svm,
    )


def train_liblinear(svm_parameters, n_iter=50):
    """
    Train SVM using liblinear (LinearSVC) with hyperparameter optimization
    """
    print("Training LibLinear model...")

    # Define parameter space
    param_distributions = {
        'C': get_reg_values(-3, 1, 5),
        'loss': ['hinge'],
        'class_weight': ['balanced', None],
        'dual': [True, False],
    }

    # Create base model
    base_model = LinearSVC(random_state=42)

    # Create RandomizedSearchCV object
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    # Fit the model
    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation score: {search.best_score_:.3f}")

    # Save model and associated objects
    model_path, scaler_path, params_path = save_model(
        search.best_estimator_,
        search.best_params_,
        'liblinear'
    )

    return search.best_estimator_, search.best_params_, search.best_score_


def train_sgd(svm_parameters, n_iter=50):
    """
    Train SVM using SGD with hyperparameter optimization
    """
    print("Training SGD model...")

    # Define parameter space
    param_distributions = {
        'alpha': get_reg_values(-6, 1, 7),
        'loss': ['hinge'],
        'penalty': ['l2'],
        'learning_rate': ['optimal', 'adaptive'],
        'class_weight': ['balanced', None],
        'average': [True, False],
    }

    # Create base model
    base_model = SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )

    # Create RandomizedSearchCV object
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    # Fit the model
    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation score: {search.best_score_:.3f}")

    # Save model and associated objects
    model_path, scaler_path, params_path = save_model(
        search.best_estimator_,
        search.best_params_,
        'sgd'
    )

    return search.best_estimator_, search.best_params_, search.best_score_


if __name__ == '__main__':
    # Train all models
    print("Starting model training...")

    # LibSVM
    time_start_libsvm = time.perf_counter()
    libsvm_model, libsvm_params, libsvm_score = train_libsvm(svm_parameters)
    time_end_libsvm = time.perf_counter()
    print(f"\nLibSVM Training Complete - Best Score: {libsvm_score:.3f}")
    print(f"Training Time: {time_end_libsvm - time_start_libsvm:.2f}s")

    # LibLinear
    time_start_liblinear = time.perf_counter()
    liblinear_model, liblinear_params, liblinear_score = train_liblinear(svm_parameters)
    time_end_liblinear = time.perf_counter()
    print(f"\nLibLinear Training Complete - Best Score: {liblinear_score:.3f}")
    print(f"Training Time: {time_end_liblinear - time_start_liblinear:.2f}s")

    # SGD
    time_start_sgd = time.perf_counter()
    sgd_model, sgd_params, sgd_score = train_sgd(svm_parameters)
    time_end_sgd = time.perf_counter()
    print(f"\nSGD Training Complete - Best Score: {sgd_score:.3f}")
    print(f"Training Time: {time_end_sgd - time_start_sgd:.2f}s")

    # Print comparative results
    print("\nModel Comparison:")
    print(f"LibSVM Score: {libsvm_score:.3f}")
    print(f"LibLinear Score: {liblinear_score:.3f}")
    print(f"SGD Score: {sgd_score:.3f}")

import numpy as np

from dataset import get_dataset_path, datasets
from main import hog_parameters
from parameters import SVM_Parameters, HOG_Parameters
from svm import load_svm
from transform import grayscale_transform, hog_transform


# def see_dataset_balance():
#     for category in ['train','test']:
#         for it, dataset in enumerate(datasets):
#             if category == 'train' and it == 1:
#                 break
#             for window in window_sizes:
#                 y = np.load(get_dataset_path(
#                     window,
#                     category,
#                     'label',
#                     dataset
#                 ))
#                 if category == 'train':
#                     print(f"Category: {category}, Window: {window}")
#                 else:
#                     print(f"Category: {category}, Dataset: {dataset}, Window: {window}")
#                 print(f"Positive: {np.sum(y)}")
#                 print(f"Negative: {y.shape[0] - np.sum(y)}\n")





if __name__ == '__main__':
    print("Testing")
    model = load_svm(
        SVM_Parameters(
            window_size=(128, 64),
            hog_parameters=HOG_Parameters(
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                block_stride=(1, 1),
holistic_derivative_mask=False
        ),

        ),
        '../computed/models'
    )

    X_raw = np.load(get_dataset_path(
        (128, 64),
        'test',
        'point',
        'INRIA'
    ),
    )[:10]
    X_test = hog_transform(grayscale_transform(X_raw), hog_parameters)
    Y_test = np.load(get_dataset_path(
        (128, 64),
        'test',
        'label',
        'INRIA'
    ))[:10]
    print(Y_test)
    print(model.predict(X_test))
    print(model.decision_function(X_test))

if __name__ == '__main__':
    liblinear = joblib.load('../computed/models/hard_libsvm.pkl')
    sgd = load_svm(svm_parameters, '../computed/models')

    table = {}
    for dataset in datasets:
        X_test_raw = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset))
        X_test = hog_transform(
            grayscale_transform(X_test_raw),
            hog_parameters
        )
        y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', dataset))

        liblinear_metric = evaluate_pedestrian_classifier(
            liblinear,
            X_test,
            y_test
        )
        # table[f'LibLinear on {dataset}'] = [
        #     liblinear_metric[score_key] for score_key in score_keys
        # ]
        table[f'LibLinear on {dataset}'] = {
            score_key: liblinear_metric[score_key] for score_key in score_keys
        }
        sgd_metric = evaluate_pedestrian_classifier(
            sgd,
            X_test,
            y_test
        )
        table[f'SGD on {dataset}'] = {
            score_key: sgd_metric[score_key] for score_key in score_keys
        }


    output_metrics_table(
        table,
        f'compare_metrics_table.svg',
        custom_detector_names=True,
        detector_col_label='Detector on Dataset',
    )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    liblinear = joblib.load('../computed/models/hard_libsvm.pkl')
    sgd = load_svm(svm_parameters, '../computed/models')

    X_test, y_test = get_concatenated_test_samples()

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mcc_f1_curve(
        liblinear,
        X_test,
        y_test,
        ax=ax,
        show_best_threshold=True
    )
    plot_mcc_f1_curve(
        sgd,
        X_test,
        y_test,
        ax=ax,
        show_best_threshold=True
    )
    plt.show()
    plt.savefig('mcc_f1_compare.svg')

if __name__ == '__main__':
    total_x = []
    total_y = []
    total_pred = []
    thresholds = []
    clf = load_svm(
        svm_parameters,
        '../computed/models'
    )
    for dataset in datasets:
        X_test_raw = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset))
        X_test = hog_transform(
            grayscale_transform(X_test_raw),
            svm_parameters.hog_parameters
        )
        y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', dataset))

        total_x.append(X_test)
        total_y.append(y_test)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mcc_f1_curve(
        clf,
        np.concatenate(total_x),
        np.concatenate(total_y),
        ax=ax,
        n_thresholds=5
    )
    # Add random performance line
    ax.axhline(y=0.5, color='0.9', linestyle='--', label='Random Performance')

    # Add worst and best performance points
    ax.plot(0.01, 0.01, 'ro', label='Worst Performance', markersize=6)
    ax.plot(0.99, 0.99, 'go', label='Best Performance', markersize=6)

    ax.set(
        title='MCC-F1 Curve',
        xlabel='F1 Score',
        ylabel='Matthew\'s Correlation Coefficient'
    )
    ax.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('exmaple.svg')

    #
    # sgd = load_svm(svm_parameters, '../computed/models')
    #
    # table = {}
    # for dataset in datasets:
    #     X_test_raw = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset))
    #     X_test = hog_transform(
    #         grayscale_transform(X_test_raw),
    #         hog_parameters
    #     )
    #     y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', dataset))
    #
    #     liblinear_metric = evaluate_pedestrian_classifier(
    #         liblinear,
    #         X_test,
    #         y_test
    #     )
    #
    #     table[f'LibLinear on {dataset}'] = {
    #         score_key: liblinear_metric[score_key] for score_key in score_keys
    #     }
    #     sgd_metric = evaluate_pedestrian_classifier(
    #         sgd,
    #         X_test,
    #         y_test
    #     )
    #     table[f'SGD on {dataset}'] = {
    #         score_key: sgd_metric[score_key] for score_key in score_keys
    #     }
    #
    #
    # fig, ax = output_metrics_table(
    #     table,
    #     custom_detector_names=True,
    #     detector_col_label='Detector on Dataset',
    # )
    #
    # plt.savefig('liblinear_vs_sgd_table.png', dpi=300)
    # plt.show()

    #

# if __name__ == '__main__':
#     for dataset in ['INRIA', 'PnPLO']:
#         for window_size in window_sizes:
#             print(f"\nsTable for {dataset} with window size {window_size}")
#             table = compute_score_table(dataset, window_size=window_size)
#             # print(table['svm_orientations_18_pixels_per_cell_(10, 10)_cells_per_block_(4, 4)_block_stride_(3, 3)_block_norm_L2-Hys_holistic_derivative_mask_False_window_(100, 50)'])
#             for it in tqdm(range(0, len(table), 40)):
#                 subtable = {key: table[key] for key in list(table.keys())[it:it+40]}
#                 output_metrics_table(subtable)
#
#                 plt.savefig(f'../tables/{dataset}/{dataset}_{window_size}_{it}.png', dpi=300, bbox_inches='tight', transparent="True", pad_inches=0)
#                 plt.close()