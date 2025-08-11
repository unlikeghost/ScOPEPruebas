import os
import pickle
import warnings
import numpy as np
from datetime import datetime
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score, log_loss,
                             fbeta_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef)


def make_report(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None) -> dict:
    """
    Generates a comprehensive performance report for results, including 
    ROC-related metrics, F1 scores, log loss, and confusion matrix evaluation.

    The function computes the receiver operating characteristic (ROC) metrics,
    such as false positive rates (fpr), true positive rates (tpr), thresholds,
    area under the ROC curve (auc_roc), and classification evaluation metrics such as
    accuracy score, F1 scores, log loss and confusion matrix from true labels and predicted values.

    Arguments:
        y_true: np.ndarray
            True binary class labels of the dataset (0 or 1).
        y_pred: np.ndarray
            Predicted binary class labels by a classifier (0 or 1).
        y_pred_proba: np.ndarray
            Predicted probabilities for each class [prob_class_0, prob_class_1], 
            used for generating the ROC curve and calculating log loss.

    Returns:
        dicta:
            A dictionary containing the computed performance metrics with the keys:
            - 'fpr': array of false positive rates.
            - 'tpr': array of true positive rates.
            - 'thresholds': array of decision thresholds for the ROC curve.
            - 'auc_roc': area under the ROC curve.
            - 'accuracy': accuracy score of the predictions.
            - 'f1_score': F1 score.
            - 'log_loss': logarithmic loss of the predictions.
            - 'confusion_matrix': confusion matrix of true vs predicted class labels.
    """
    
    y_pred_proba_array = np.array(y_pred_proba)
    
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba_array[:, 1])
    except Exception as e:
        print(f"Warning: Error calculando ROC curve: {e}")
        fpr, tpr, thresholds = None, None, None
    
    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba_array[:, 1])
    except Exception as e:
        print(f"Warning: Error calculando ROC/AUC: {e}")
        auc_roc = 0.5
    
    try:
        auc_pr = average_precision_score(y_true, y_pred_proba_array[:, 1])
    except Exception as e:
        print(f"Warning: Error calculando AUC-PR: {e}")
        auc_pr = 0.0
    
    try:
        f1 = f1_score(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Error calculando F1 score: {e}")
        f1 = 0.0
    
    try:
        f2 = fbeta_score(y_true, y_pred, beta=2)
    except Exception as e:
        print(f"Warning: Error calculando F2 score: {e}")
        f2 = 0.0  
    
    try:
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy =  balanced_accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Error calculando accuracy: {e}")
        accuracy = 0.0
        balanced_accuracy = 0.0
    
    try:
        logloss = log_loss(y_true, y_pred_proba_array)
    except Exception as e:
        print(f"Warning: Error calculando log loss: {e}")
        logloss = 1.0
    
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Error calculando confusion matrix: {e}")
        conf_matrix = None
    
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Error calculando  Matthews correlation coefficient: {e}")
        mcc = None

    this_data: dict = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'f2': f2,
        'log_loss': logloss,
        'confusion_matrix': conf_matrix,
        'mcc': mcc,
    }
    
    if save_path:
        try:
            os.makedirs(save_path, exist_ok=True)
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            filepath = os.path.join(save_path, filename)
        
            with open(filepath, 'wb') as f:
                pickle.dump(this_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Report saved to {filepath}")
        
        except Exception as e:
            warnings.warn(f"Failed to save report to {save_path}: {e}")

    return this_data