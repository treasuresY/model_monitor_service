from sklearn import metrics

if __name__ == "__main__":

    y_true = [1, -1]
    y_pred = [1, 1]

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    print(precision, recall, thresholds)

    print(metrics.class_likelihood_ratios(y_true, y_pred))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    print(fpr, tpr, thresholds)

    fpr, fnr, thresholds = metrics.det_curve(y_true, y_pred)
    print(fpr, fnr, thresholds)
