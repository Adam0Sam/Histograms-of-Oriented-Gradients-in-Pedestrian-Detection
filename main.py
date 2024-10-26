import os

import joblib
import numpy as np
from matplotlib import pyplot as plt
from networkx import difference
from skimage.metrics import contingency_table
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import plot_tree
from tqdm import tqdm

from dataset import get_dataset_path, datasets
from evaluate import evaluate_pedestrian_classifier, print_evaluation_summary, score_keys, compute_score_table, \
    compute_score_table_main, get_k_rows
from mcnemar import construct_mcnemar_table
from parameters import HOG_Parameters, SVM_Parameters
from plot import output_metrics_table, plot_mcc_f1_curve, parse_detector_name
from svm import load_svm
from transform import hog_transform, grayscale_transform
from variables import holistic_derivative_masks, window_sizes, get_model_count, iterate_model_parameters
from statsmodels.stats.contingency_tables import mcnemar

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


if __name__ == '__main__':

    best_mcc_name = [t[0] for t in get_k_rows('mcc','total', top=True)][0]
    worst_mcc_name = [t[0] for t in get_k_rows('mcc','total', top=True)][1]

    best_mcc = joblib.load(f'../computed/models/{best_mcc_name}.pkl')
    worst_mcc = joblib.load(f'../computed/models/{worst_mcc_name}.pkl')

    best_mcc_params = parse_detector_name(best_mcc_name, return_object=True)
    worst_mcc_params = parse_detector_name(worst_mcc_name, return_object=True)

    best_mcc_x, y = get_concatenated_test_samples(best_mcc_params)
    worst_mcc_x, _ = get_concatenated_test_samples(worst_mcc_params)

    best_mcc_pred = best_mcc.predict(best_mcc_x)
    worst_mcc_pred = worst_mcc.predict(worst_mcc_x)
    
    mcnemar_score = construct_mcnemar_table(y, best_mcc_pred, worst_mcc_pred)
    print(f"{best_mcc_name}\n vs \n{worst_mcc_name}\n\n")
    print(mcnemar(mcnemar_score))
