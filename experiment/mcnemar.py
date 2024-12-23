import numpy as np


def construct_mcnemar_table(
        y_true,
        model_1_pred,
        model_2_pred
):
    '''
    Constructs a 2x2 contingency table for McNemar's test based on the predictions of two models.

    Parameters:
    -----------
    y_true : list or array-like
        The true class labels for the test set.

    model_1_pred : list or array-like
        The predicted class labels from the first model.

    model_2_pred : list or array-like
        The predicted class labels from the second model.

    Returns:
    --------
    contingency_table : np.ndarray
        A 2x2 numpy array that represents the contingency table:
            [[a, b], [c, d]]
        where:
        - a = Both models correctly classify the instance.
        - b = Model 1 is correct, but Model 2 is incorrect.
        - c = Model 1 is incorrect, but Model 2 is correct.
        - d = Both models incorrectly classify the instance.
    '''
    a = b = c = d = 0

    for i in range(len(y_true)):
        model_1_correct = (model_1_pred[i] == y_true[i])
        model_2_correct = (model_2_pred[i] == y_true[i])

        if model_1_correct and model_2_correct:
            a += 1
        elif model_1_correct and not model_2_correct:
            b += 1
        elif not model_1_correct and model_2_correct:
            c += 1
        else:
            d += 1
    contingency_table = np.array([[a, b], [c, d]])
    return contingency_table
