#!/usr/bin/python3
import random
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from argparse import ArgumentParser
from sklearn.externals import joblib

import Converters
from Converters.converters import converter_1
from Model.AutoEncoder import Autoen
from Model.CompressedAutoEncoder import compressing_autoencoder
from Model.OneClassSVM import OneClassSVM
from Model.SVM import SVM
from measures.metrics import outputresult, feedbackdata

import timeit
import os

filedirect = '/home/khuhro/FrameWork'

opter = ''
losser = ''


class Framework:

    def __init__(self, data, dataname, freshlist, iterator, runs):
        self.data = data
        self.dataname = dataname

        self.freshlist = freshlist
        self.iterator = iterator
        self.runs = runs

    def adder(self, result, filename, alg, Optimizer, loss):
        '''

        :param result:  Takes in results of parameters
        :param filename: Name of Dataset and Alogoritm
        :return: Final average result of all the runs
        '''
        import os
        import pandas as pd
        global filedirect

        direct = filedirect
        print(direct)
        os.chdir(direct)
        direct = direct + '/savedresult'

        datalist = list((os.listdir()))
        direct2 = direct + '/' + filename + '.pkl'
        DatasetNames = []

        for items in datalist:
            Name, _ = os.path.splitext(items)
            DatasetNames.append(Name)

        freshlist = self.freshlist
        iterator = self.iterator
        '''
        if filename not in DatasetNames:
            joblib.dump(freshlist, direct2)
        '''
        listoflist = []

        # listback = joblib.load(direct2)
        for i in range(7):
            freshlist[alg][i][iterator] = result[i]

        print(freshlist)
        if iterator == (self.runs - 1):

            for i in range(7):
                listoflist.append(np.mean(freshlist[alg][i]))

            listoflist.append(np.std(freshlist[alg][4]))
            listoflist.append(np.std(freshlist[alg][5]))

            listoflist.append(str(Optimizer))
            listoflist.append(str(loss))
            joblib.dump(listoflist, direct2)

    # adding file path folder extension to the file path

    def Iocsvm(self):

        alg = 0
        data = self.data

        X_train, X_val, X_test, y_train, y_val, y_test = converter_1(data,
                                                                     onClassClassification=True,
                                                                     shuffle=False, test_size=0.25,
                                                                     validation_size=0.25)

        start = timeit.default_timer()
        loader = OneClassSVM(X_train, X_val, X_test, y_train, y_val, y_test)

        y_test, y_pred = loader.prediction()

        end = timeit.default_timer()
        totaltime = round(end - start, 2)
        Name = 'OneClassSVM' + '-' + self.dataname

        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]
        self.adder(results, Name, alg)

    def Ae(self):
        global opter
        global losser
        # Initial function to perform grid search
        if self.iterator == 0:
            Optimizers = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'SGD']

            loss = ['binary_crossentropy', 'categorical_crossentropy','cosine_similarity',
                    'mean_absolute_error',
                    'mean_absolute_percentage_error', 'mean_squared_error', 'mean_squared_logarithmic_error', 'poisson'
                    ]
            test2 = 0

            for elements in range(len(Optimizers)):
                for elemento in range(len(loss)):
                    alg = 1
                    data = self.data
                    train, valid, validB, testB = converter_1(data, AE=True)

                    print(Optimizers[elements], 'Optimizer')
                    print(loss[elemento], ' loss')
                    loader = Autoen(train, valid, validB, testB, Optimizers[elements], loss[elemento])

                    y_test, y_pred = loader.Ae()

                    _, _, _, _, num, num2 = feedbackdata(y_test, y_pred)

                    if num2 <= 1:
                        num2 = 1
                    teste = num * (np.power(num, (1 / num2)))
                    if teste > test2:
                        opter = Optimizers[elements]
                        losser = loss[elemento]
                        test2 = teste

        print(opter, ' PRINTINGGGGG OPTIMIZZERRRR')
        print(losser, 'PRINTTINGG LOSSS')
        alg = 1
        data = self.data
        train, valid, validB, testB = converter_1(data, AE=True)

        start = timeit.default_timer()
        loader = Autoen(train, valid, validB, testB, opter, losser)

        y_test, y_pred = loader.Ae()
        end = timeit.default_timer()
        totaltime = round(end - start, 2)
        Name = 'AutoEncoder' + '-' + self.dataname
        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]
        self.adder(results, Name, alg, opter, losser)

        print(self.iterator, "HEEEEEEEREEEEEEEEEEEEEE ITERATOR 22222")

    def C_AE(self):

        alg = 2
        data = self.data
        train, validB, testB = converter_1(data, cAE=True)

        start = timeit.default_timer()
        loader = compressing_autoencoder(train, validB, testB)

        loader.loading_data()
        y_test, y_pred = loader.test()
        end = timeit.default_timer()
        totaltime = round(end - start, 2)

        Name = 'Compressed_AutoEncoder' + '-' + self.dataname
        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]
        self.adder(results, Name, alg)

        print(self.iterator, "HEEEEEEEREEEEEEEEEEEEEE ITERATOR 33333333333333")

    def SVM(self):

        alg = 3
        data = self.data

        X_train, X_test, y_train, y_test = converter_1(data, SVM=True)

        start = timeit.default_timer()

        loader = SVM(X_train, X_test, y_train, y_test)
        y_test, y_pred = loader.svm()
        end = timeit.default_timer()
        totaltime = round(end - start, 2)

        Name = 'SVM' + '-' + self.dataname
        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]

        self.adder(results, Name, alg)
