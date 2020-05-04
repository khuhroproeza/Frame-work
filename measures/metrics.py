import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def outputresult(raw, pred):
    raw = raw
    pred = pred
    conf_matrix = confusion_matrix(raw, pred)
    precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
        raw, pred)
    accuracy = accuracy_score(raw, pred)
    con = conf_matrix
    # print(accuracy)
    print("\n")
    # print("Confusion Matrix", "\n", con, "\n")
    print("Precision Normal : ", round(precision_N[0], 2)*100, " | ", "Precision Anomaly : ", round(precision_N[1], 2)*100,
          "\n")
    print("Recall Normal    : ", round(recall_N[0], 2)*100, " | ", "Recall Anomaly    : ", round(recall_N[1], 2)*100, "\n")
    print("Fscore Normal    : ", round(fscore_N[0], 2)*100, " | ", "Fscore Anomaly    : ", round(fscore_N[1], 2)*100)


def detection_rat(tn, fp, fn, tp):
    detection_rate = (tp + fn)

    detection_rate = np.true_divide(tp, detection_rate)
    return detection_rate * 100


def detection_rate_false_alarm_rate(y_test, y_pred):
    # from mlxtend.evaluate import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    #print('\nConfusion Metrics \n \n', conf_matrix)
    # print('\n')
    tn, fp, fn, tp = conf_matrix.ravel()
    print(tn, fp, tp, fn)
    detection_rate = round(detection_rat(tn, fp, fn, tp), 2)
    false_positive_rate = round((np.true_divide(fp, (tn + fp)) * 100), 2)
    # print(detection_rate,false_positive_rate)
    return detection_rate, false_positive_rate


def feedbackdata(y_test, y_pred):
    # from mlxtend.evaluate import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print('\nConfusion Metrics \n \n', conf_matrix)
    # print('\n')
    tn, fp, fn, tp = conf_matrix.ravel()

    #print(tn, fp, tp, fn)
    detection_rate = round(detection_rat(tn, fp, fn, tp), 2)
    false_positive_rate = round((np.true_divide(fp, (tn + fp)) * 100), 2)
    # print(detection_rate,false_positive_rate)
    return int(tn), int(fp), int(fn), int(tp), detection_rate, false_positive_rate
