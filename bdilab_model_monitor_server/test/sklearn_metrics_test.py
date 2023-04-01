from sklearn import metrics
import numpy as np

if __name__ == "__main__":
    y_true1 = [[0,1], [1,0]]
    y_pred1 = [[1,0], [1,0]]
    y_true2 = [1, 2]
    y_pred2 = [2, 2]
    # print(metrics.accuracy_score(y_true1, y_pred1))
    # print(metrics.accuracy_score(y_true2, y_pred2))

    # F1 = 2 * (precision * recall) / (precision + recall)
    # print(metrics.f1_score(y_true1, y_pred1, average=None, zero_division=0))
    # print(metrics.f1_score(y_true2, y_pred2, average=None, zero_division=0))
    # print(metrics.f1_score(y_true1, y_pred1, average='macro', zero_division=0))
    # print(metrics.f1_score(y_true2, y_pred2, average='macro', zero_division=0))
    # print(metrics.f1_score(y_true1, y_pred1, average='weighted', zero_division=0))
    # print(metrics.f1_score(y_true2, y_pred2, average='weighted', zero_division=0))
    # print(metrics.f1_score(y_true1, y_pred1, average='micro', zero_division=0))
    # print(metrics.f1_score(y_true2, y_pred2, average='micro', zero_division=0))

    # tp / (tp + fn)
    # print(metrics.precision_score(y_true1, y_pred1, average=None, zero_division=0))
    # print(metrics.precision_score(y_true2, y_pred2, average=None, zero_division=0))
    # print(metrics.precision_score(y_true1, y_pred1, average='macro', zero_division=0))
    # print(metrics.precision_score(y_true2, y_pred2, average='macro', zero_division=0))

    # tp / (tp + fp)
    # print(metrics.recall_score(y_true1, y_pred1, average=None, zero_division=0))
    # print(metrics.recall_score(y_true2, y_pred2, average=None, zero_division=0))
    # print(metrics.recall_score(y_true1, y_pred1, average='macro', zero_division=0))
    # print(metrics.recall_score(y_true2, y_pred2, average='macro', zero_division=0))

    # print(metrics.log_loss(y_true1, y_pred1))
    # print(metrics.log_loss(y_true2, y_pred2))

    # print(metrics.multilabel_confusion_matrix(y_true1, y_pred1))
    # print(metrics.multilabel_confusion_matrix(y_true2, y_pred2))
    # print(metrics.confusion_matrix(y_true1, y_pred1))
    # print(metrics.confusion_matrix(y_true2, y_pred2))
    # y_t = [1]
    # y_p = [1]
    # print(metrics.multilabel_confusion_matrix(y_t, y_p))
    # print(metrics.confusion_matrix(y_t, y_p))

    # print(metrics.roc_auc_score(y_true1, y_pred1, average=None))
    # print(metrics.roc_auc_score(y_true2, y_pred2, average=None))
    # print(metrics.roc_auc_score(y_true1, y_pred1, average='weighted'))
    # print(metrics.roc_auc_score(y_true2, y_pred2, average='weighted'))

    # binary classification case
    # tp / (tp + fp)
