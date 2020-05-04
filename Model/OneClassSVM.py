from datetime import datetime

import numpy as np
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from measures.metrics import feedbackdata
from measures.metrics import outputresult, feedbackdata


class OneClassSVM:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, kernel, gamma, nu, showModelSummary=False,
                 threshold_fn_percentage=0.10):
        print("initializing ..")
        self.successMsg = '\x1b[0;30;42m' + 'Success!' + '\x1b[0m'
        # data params
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train.astype(np.float))
        X_val = min_max_scaler.transform(X_val.astype(np.float))
        X_test = min_max_scaler.transform(X_test.astype(np.float))
        self.gamma = gamma
        self.kernel = kernel
        self.nu = float(nu)
        self.threshold_fn_percentage = threshold_fn_percentage
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        # model params
        self.showModelSummary = showModelSummary
        self.load_model()

    def load_model(self):
        clf = svm.OneClassSVM(gamma=self.gamma, kernel=self.kernel, nu=self.nu)
        print('\ntraining the classifier  start time: ', str(datetime.now()))
        print('\n', clf)
        self.clf = clf.fit(self.X_train)
        self.calculate_error_threshold()

    def calculate_error_threshold(self):
        val_score = self.clf.score_samples(self.X_val)
        min_max_scaler = preprocessing.MinMaxScaler()
        val_score = min_max_scaler.fit_transform(val_score.reshape(-1, 1))
        acceptable_n_FP = self.threshold_fn_percentage * len(self.y_val)
        threshold = 0
        besterthreshold = 0
        best_fscore_W = 0
        while (threshold <= 1):

            # print ('**************************')
            # print (threshold)
            threshold += 0.005
            y_pred = [1 if e > threshold else 0 for e in val_score]
            # y_Pred = np.array(y_pred)
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
            conf_matrix = confusion_matrix(self.y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()
            # print(conf_matrix)
            # print("tn: " , tn)
            # print("fp: " ,fp)

            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

            conf_matrix = confusion_matrix(self.y_val, y_pred, labels=[0, 1])

            precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
                self.y_val, y_pred, average='binary', pos_label=0)
            precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
                self.y_val, y_pred, average='binary', pos_label=1)
            precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
                self.y_val, y_pred, average='weighted')
            tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(self.y_val, y_pred)

            from sklearn import metrics

            import pandas as pd
            yval = pd.DataFrame(self.y_val)
            import numpy as np
            from sklearn import metrics

            fpr, tpr, _ = metrics.roc_curve(self.y_val, y_pred)
            sacore = metrics.auc(fpr, tpr)
            print(sacore)
            if fscore_A > best_fscore_W:
                best_fscore_W = fscore_A
                besterthreshold = threshold

            # print(predx.count(0))

            # print(yval)
            # print(yval.describe())

            # if fscore_A> recall_A_best:

        self.threshold = besterthreshold

        self.prediction()
        # return threshold

    def prediction(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        test_score = min_max_scaler.fit_transform(self.clf.score_samples(self.X_test).reshape(-1, 1))

        y_pred = [1 if e >= self.threshold else 0 for e in test_score]
        self.y_pred = y_pred
        conf_matrix = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        print(conf_matrix, "HEEEEEEEEEEEEEREEEEEEEEEE RESULT")
        return y_pred, self.y_test


best_resultssvm = 0
bestresultssvmlist = []


def SVMhyp(X_train, X_val, X_test, y_train, y_val, y_test, threshold_fn_percentage=0.10):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train.astype(np.float))
    X_val = min_max_scaler.transform(X_val.astype(np.float))
    X_test = min_max_scaler.transform(X_test.astype(np.float))

    threshold_fn_percentage = threshold_fn_percentage
    X_train = X_train
    X_val = X_val
    X_test = X_test
    y_train = y_train
    y_val = y_val
    y_test = y_test
    # model params
    nu = Categorical(categories=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], name='nu')
    kernel = Categorical(
        categories=['linear', 'poly', 'rbf'],
        name='kernel')
    # gamma = Integer(low=1, high=5, name='gamma')
    gamma = Real(low=1e-3, high=2, prior='log-uniform',
                 name='gamma')
    dimensions = [kernel, gamma, nu]

    @use_named_args(dimensions=dimensions)
    def svmHyper(kernel, gamma, nu):
        nu = float(nu)
        # global best_resultssvm
        clf = svm.OneClassSVM(gamma=gamma, kernel=kernel, nu=nu)
        print('\ntraining the classifier  start time: ', str(datetime.now()))
        print('\n', clf)
        clf = clf.fit(X_train)
        val_score = clf.score_samples(X_val)
        threshold = 0
        best_threshold = 0
        acceptable_n_FP = threshold_fn_percentage * len(y_train)
        print("\nacceptable_n_FP: ", acceptable_n_FP)
        TN = 0
        FP = 0
        print("\ncalculating threshold......")
        while (threshold <= 1):
            # print ('**************************')
            # print (threshold)
            threshold += 0.005
            y_pred = [1 if e > threshold else 0 for e in val_score]
            y_Pred = np.array(y_pred)
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
            conf_matrix = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()
            # print(conf_matrix)
            # print("tn: " , tn)
            # print("fp: " ,fp)
            if fp < acceptable_n_FP:
                break
        best_threshold = threshold
        y_pred = [1 if e >= best_threshold else 0 for e in val_score]

        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            y_val, y_pred, average='weighted')
        global best_resultssvm
        # global bestresultssvmlist
        print(fscore_W)
        if fscore_W > best_resultssvm:
            # Save the new model to harddisk.
            # autoencoder.save(path_best_model)learning_rate,  activation_function,  optimizer, loss ,num_dense_layers, batch_size
            bestresultssvmlist.append(kernel)
            bestresultssvmlist.append(gamma)
            bestresultssvmlist.append(nu)
            # Update the classification accuracy.
            best_resultssvm = fscore_W

        # accuracy, we need to negate this number so it can be minimized.
        return -fscore_W

    default_parameters = ['rbf', 1e-3, '0.2']
    search_result = gp_minimize(func=svmHyper, dimensions=dimensions, acq_func='EI', n_calls=50, x0=default_parameters)

    return bestresultssvmlist[-3:]
