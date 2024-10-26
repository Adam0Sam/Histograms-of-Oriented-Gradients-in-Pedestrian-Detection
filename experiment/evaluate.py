import os
import warnings
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc, log_loss, recall_score, precision_score, f1_score, \
    precision_recall_curve, confusion_matrix, matthews_corrcoef

from dataset import get_dataset_path, datasets
from parameters import HOG_Parameters, SVM_Parameters
from svm import load_svm
from transform import hog_transform, grayscale_transform
from variables import iterate_model_parameters, get_model_count

score_keys = ['mcc', 'accuracy', 'f1', 'fppw', 'auc_roc', 'average_precision']
score_index_map = {key: i for i, key in enumerate(score_keys)}

def evaluate_pedestrian_classifier(model, X_test, y_test):
    """
    Evaluate a binary classifier for pedestrian detection using multiple metrics.

    Parameters:
    -----------
    model : trained classifier object
        Must implement predict() and predict_proba() or decision_function()
    X_test : array-like
        Test features
    y_test : array-like
        True labels (0 for non-pedestrian, 1 for pedestrian)

    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    metrics = {}

    # If probabilities not available, use decision function
    y_scores = model.decision_function(X_test)
    # Normalize scores to [0,1] range for better interpretability
    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    y_pred = model.predict(X_test)

    # Basic classification metrics
    metrics['accuracy'] = np.mean(y_pred == y_test)

    # Confusion matrix and derived metrics
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['true_negatives'] = cm[0, 0]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_positives'] = cm[1, 1]

    # Precision, Recall, F1
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred)

    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    # Class-wise metrics
    metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # True Negative Rate
    metrics['fall_out'] = cm[0, 1] / (cm[0, 0] + cm[0, 1])  # False Positive Rate
    metrics['miss_rate'] = cm[1, 0] / (cm[1, 0] + cm[1, 1])  # False Negative Rate

    if y_scores is not None:
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
        metrics['pr_curve'] = {
            'precision': precision,
            'recall': recall,
            'thresholds': pr_thresholds
        }
        metrics['average_precision'] = average_precision_score(y_test, y_scores)

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
        metrics['roc_curve'] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds
        }
        metrics['auc_roc'] = auc(fpr, tpr)

    # Add some practical metrics
    total_windows = len(y_test)
    metrics['fppw'] = metrics['false_positives'] / total_windows

    return metrics


def print_evaluation_summary(metrics):
    """
    Print a human-readable summary of the evaluation metrics.

    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics returned by evaluate_pedestrian_classifier
    """
    print("Classification Performance Summary")
    print("=================================")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("\nConfusion Matrix:")
    print(f"TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}")
    print(f"FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}")
    print("\nKey Metrics:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall (Detection Rate): {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"False Positives per Window: {metrics['fppw']:.4f}")

    if 'auc_roc' in metrics:
        print(f"\nAUC-ROC: {metrics['auc_roc']:.3f}")
    if 'average_precision' in metrics:
        print(f"Average Precision: {metrics['average_precision']:.3f}")
    if 'log_loss' in metrics:
        print(f"Log Loss: {metrics['log_loss']:.3f}")

def get_k_rows(metric, dataset, row_count=10, top=True):
    if metric not in score_keys:
        raise ValueError(f"Invalid Metric: {metric}")
    if dataset not in datasets and dataset != 'total':
        raise ValueError(f"Invalid Dataset: {dataset}")

    score_map = {}

    def get_score(svm_parameters):
        score_file_name = f"../computed/scores/{dataset}/{svm_parameters.get_svm_name()}.npy"
        if os.path.exists(score_file_name):
            score = np.load(score_file_name)
            score_map[svm_parameters.get_svm_name()] = score
        else:
            print(f"Score for {svm_parameters.get_svm_name()} not found")

    iterate_model_parameters(get_score)

    sorted_scores = sorted(score_map.items(), key=lambda x: x[1][score_keys.index(metric)], reverse=top)
    top_k_rows = sorted_scores[:row_count]
    return [(row_name, scores[score_keys.index(metric)]) for row_name, scores in top_k_rows]


def compute_score_table(dataset, window_size=None, cheap=None, overwrite=False):
    table = {}
    current_row = 0
    total = get_model_count(cheap=cheap)
    all_exist = True
    def compute_row(svm_parameters):
        nonlocal current_row
        nonlocal table
        nonlocal all_exist

        if window_size is not None and svm_parameters.window_size != window_size:
            return

        score_file_name = f"../computed/scores/{dataset}/{svm_parameters.get_svm_name()}.npy"
        if os.path.exists(score_file_name) and not overwrite:
            if all_exist is False:
                print(f"EXISTS: Score for {svm_parameters.get_svm_name()}")
            current_row += 1
            table[svm_parameters.get_svm_name()] = np.load(score_file_name)
            return
        else:
            print(f"Computing {current_row}/{total}: {svm_parameters.get_svm_name()}")
            all_exist = False
            model = load_svm(svm_parameters, '../computed/models')
            X = hog_transform(
                grayscale_transform(np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset))),
                              svm_parameters.hog_parameters)
            y = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label', dataset))
            metric = evaluate_pedestrian_classifier(model, X, y)
            score = [metric[key] for key in score_keys]
            table[svm_parameters.get_svm_name()] = score
            np.save(score_file_name, score)
            current_row += 1

    iterate_model_parameters(compute_row, cheap=cheap)
    if all_exist:
        print("All scores exist")
    return table


def compute_score_table_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--cheap", action="store_true", default=None)
    parser.add_argument("--expensive", dest='cheap', action="store_false")
    args = parser.parse_args()
    compute_score_table(args.dataset,cheap=args.cheap,overwrite=True)

def get_dimensions(hog_parameters: HOG_Parameters, window_size):
    dimensions = ((window_size[0] - hog_parameters.pixels_per_cell[0] * (hog_parameters.cells_per_block[0] - 1)) // (hog_parameters.block_stride[0] * hog_parameters.pixels_per_cell[0])) * \
                 ((window_size[1] - hog_parameters.pixels_per_cell[1] * (hog_parameters.cells_per_block[1] - 1)) // (hog_parameters.block_stride[1] * hog_parameters.pixels_per_cell[1])) * \
                 hog_parameters.cells_per_block[0] * hog_parameters.cells_per_block[1] * hog_parameters.orientations
    return dimensions

def get_dimension_map():
    dimension_map = {}
    def get_dimensions(svm_parameters: SVM_Parameters):
        dimensions = get_dimensions(
            svm_parameters.hog_parameters,
            svm_parameters.window_size
        )
        dimension_map[dimensions] = (
            svm_parameters.hog_parameters.get_hog_name(), svm_parameters.window_size
        )

    iterate_model_parameters(get_dimensions)
    return dimension_map

