import re
import tracemalloc
from resource import getrusage, RUSAGE_SELF

from matplotlib import rc, pyplot as plt
from mcc_f1 import mcc_f1_curve
from mcc_f1._plot.base import _get_response
from scipy import stats as st
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, matthews_corrcoef
from dataset import get_dataset_path, datasets
from evaluate import score_index_map, score_keys, get_dimension_map
from hog import HOG_Parameters
from svm import SVM_Parameters, load_svm
from transform import grayscale_transform, hog_transform, hog_transform_cache
from mcc_f1_curve_display import MCCF1CurveDisplay

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

def dimension_computation_corr_main(
        compute_times,
        title,
        ylabel,
        custom_name='correlation_with_best_fit'
    ):
    # dimension_time_map = measure_hog_computation_time()
    #
    # dimensions = np.array(list(dimension_time_map.keys()))
    # compute_times = np.array(list(dimension_time_map.values()))
    # np.savez('dimension_time_map.npz', dimensions=dimensions, compute_times=compute_times)

    dimensions = np.load('dimension_time_map.npz')['dimensions']

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(dimensions, compute_times, color='blue', label='Data Points')

    # Calculate the line of best fit
    slope, intercept = np.polyfit(dimensions, compute_times, 1)
    line_of_best_fit = slope * dimensions + intercept

    # Plot the line of best fit
    plt.plot(dimensions, line_of_best_fit, color='red', label='Line of Best Fit')

    # Add titles and labels
    plt.title(title)
    plt.xlabel('HOG Feature Vector Dimensionality')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the figure
    # plt.savefig('correlation_with_best_fit.svg')
    plt.savefig(f'{custom_name}.svg')
    plt.show()



def plot_kde_dimensions():
    dimension_map = get_dimension_map()
    dimensions_sorted = sorted(dimension_map.keys())
    kde = st.gaussian_kde(dimensions_sorted)
    kde_values = kde(dimensions_sorted)
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_sorted, kde_values, color='blue', label='KDE (Trend)')
    plt.scatter(dimensions_sorted, np.zeros_like(dimensions_sorted), color='red', marker='o', label='Discrete Values')
    plt.title('Dimension Distribution with KDE')
    plt.xlabel('Dimension Values')
    plt.ylabel('Density / Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dimension_distribution.svg')

    max_density_index = np.argmax(kde_values)
    max_density = kde_values[max_density_index]
    max_density_dimension = dimensions_sorted[max_density_index]
    print(f"Highest Dimension Value: {max(dimensions_sorted)}")
    print(f"Max Density: {max_density}")
    print(f"Max Density Dimension: {max_density_dimension}")


def plot_roc_curve(svm_parameters: SVM_Parameters):

    svm = load_svm(svm_parameters, '../../saved_models')

    fig, ax = plt.subplots(figsize=(6, 6))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, dataset in enumerate(datasets):
        plot_chance_level = True if i == 0 else False
        X_gray= grayscale_transform(np.load(get_dataset_path(svm_parameters.window_size, 'test', 'point', dataset)))
        X_test = hog_transform(X_gray,svm_parameters.hog_parameters)
        y_test = np.load(get_dataset_path(svm_parameters.window_size, 'test', 'label',dataset))
        viz = RocCurveDisplay.from_estimator(
            svm,
            X_test,
            y_test,
            name=dataset,
            ax=ax,
            alpha=0.5,
            plot_chance_level=plot_chance_level
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        lw=2,
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        color='b',
        alpha=0.8
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability",
    )
    ax.legend(loc="lower right")
    plt.show()
    

def plot_mcc_f1_curve(estimator, X, y, *, sample_weight=None,
                      response_method="auto", name=None, ax=None,
                      pos_label=None, n_thresholds=0, show_best_threshold=False, **kwargs):
    """Plot MCC-F1 curve with threshold values.

    Parameters
    ----------
    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    y : array-like of shape (n_samples,)
        Target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    response_method : {'predict_proba', 'decision_function', 'auto'} \
    default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.

    name : str, default=None
        Name of MCC-F1 Curve for labeling. If `None`, use the name of the
        estimator.

    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.

    pos_label : str or int, default=None
        The class considered as the positive class when computing the metrics.
        By default, `estimators.classes_[1]` is considered as the positive
        class.

    n_thresholds : int, default=5
        Number of threshold values to display on the curve.

    show_best_threshold : bool, default=False
        Whether to show the best threshold in the legend.
    """
    y_pred, pos_label = _get_response(
        X, estimator, response_method, pos_label=pos_label)

    mcc, f1, thresholds = mcc_f1_curve(y, y_pred, pos_label=pos_label,
                                       sample_weight=sample_weight)
    mcc_f1 = None

    name = estimator.__class__.__name__ if name is None else name

    viz = MCCF1CurveDisplay(
        f1=f1,
        mcc=mcc,
        thresholds=thresholds,
        mcc_f1=mcc_f1,
        estimator_name=name,
        pos_label=pos_label,
    )
    ax.set(
        title='MCC-F1 Curve',
        xlabel='F1 Score',
        ylabel='Matthew’s Correlation Coefficient'
    )
    ax.legend(loc="lower right")
    return viz.plot(ax=ax, name=name, n_thresholds=n_thresholds, show_best_threshold=show_best_threshold, **kwargs)


def output_metrics_table(metrics_dict, custom_detector_names=False, detector_col_label=None):
    """
    Create a table of evaluation metrics for multiple detectors and save it as an SVG image.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary where keys are detector names and values are another dict with metrics.
        Expected metrics: accuracy, precision, recall, f1, specificity, fppw, auc_roc,
                        average_precision, log_loss
    filename : str
        The filename to save the table (should end in .svg)
    """
    # Define metrics to show and their display names
    table_metric_names = {
        'mcc': 'MCC',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'fppw': 'FPPW',
        'auc_roc': 'AUC-ROC',
        'average_precision': 'AP',
    }

    # Define order of metrics
    score_keys = list(table_metric_names.keys())
    # Extract detector names and metrics
    detectors = list(metrics_dict.keys())
    # Prepare the table data
    table_data = []
    for detector in detectors:
        if custom_detector_names:
            row = [detector]
        else:
            row = [get_svm_params(detector).get_svm_name()]
        for metric in score_keys:
            try:

                scores = metrics_dict[detector]

                if isinstance(scores, dict):
                    value = scores[metric]
                # is list or <class 'numpy.ndarray'>
                elif isinstance(scores, (np.ndarray, )) or isinstance(scores, list):
                    value = metrics_dict[detector][score_index_map[metric]]
                else:
                    raise ValueError("Invalid scores format")

                # Format very small numbers in scientific notation
                if isinstance(value, (int, float)) and abs(value) < 0.001:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{float(value):.3f}"
                row.append(formatted_value)
            except (KeyError, ValueError):
                raise ValueError(f"Invalid scores format for {detector}, {metric}, {scores}")
        table_data.append(row)

    # Calculate figure size based on content
    n_rows = len(table_data)
    n_cols = len(score_keys) + 1

    # Fixed base height per row (in inches)
    base_row_height = 0.4

    # Calculate dimensions
    fig_width = max(8, n_cols * 1.5)
    fig_height = (n_rows + 1) * base_row_height  # +1 for header row

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Create column labels
    col1_label = detector_col_label if detector_col_label is not None and custom_detector_names else "$\omega$-$(c_h,c_w)$-$(b_h,b_w)$-$(s_h,s_w)$-hdm"
    col_labels = [col1_label] + [
        table_metric_names[metric] for metric in score_keys
    ]
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )

    # Style the table
    fixed_height = 1.0 / (n_rows + 1)  # Uniform height for all rows
    for (i, j), cell in table.get_celld().items():
        # Header row styling
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

        # First column styling (detector names)
        if j == 0:
            cell.set_width(0.3)

        if i == 0 and j == 0:
            cell.set_text_props(weight='bold')

        # Set fixed cell height
        cell.set_height(fixed_height)

        # Set cell border properties
        cell.set_linewidth(1)
        cell.set_edgecolor('#cccccc')

    # Set font properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    return fig, ax


def plot_measurements(time_map, memory_map):
    """Helper function to create visualizations of the measurements"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract times and dimensions
    dimensions = list(time_map.keys())

    print(memory_map)

    times = [time_map[dimension] for dimension in dimensions]
    peak_memories = [data['peak_memory'] for data in memory_map.values()]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot computation times
    sns.barplot(x=dimensions, y=times, ax=ax1)
    ax1.set_title('Computation Time by Dimensions')
    ax1.set_xlabel('Dimensions')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)

    # Plot memory usage
    sns.barplot(x=dimensions, y=peak_memories, ax=ax2)
    ax2.set_title('Peak Memory Usage by Dimensions')
    ax2.set_xlabel('Dimensions')
    ax2.set_ylabel('Memory (MB)')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig