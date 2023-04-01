from sklearn import metrics

if __name__ == "__main__":

    y_true = [1, 5]
    y_pred = [1, 4]
    y_true1 = [[0,1], [1,0]]
    y_pred1 = [[1,0], [1,0]]

    print(metrics.balanced_accuracy_score(y_true, y_pred))
    print(metrics.cohen_kappa_score(y_true, y_pred))

