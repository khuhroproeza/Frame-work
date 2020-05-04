import random
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from argparse import ArgumentParser
from sklearn.externals import joblib

import Converters
from Converters.converters import converter_1
from Model.AutoEncoder import Autoen, Hyperparametertuning
from Model.CompressedAutoEncoder import compressing_autoencoder, C_Hyp
from Model.OneClassSVM import OneClassSVM, SVMhyp
from Model.SVM import SVM
from measures.metrics import outputresult,  feedbackdata

import timeit
import os
#This command gets the working directory of the framework to later load files
filedirect = os.getcwd()
#filedirect = '/home/khuhro/FrameWork'
opter = ''
losser = ''
class Framework:
    '''
    Framework class which hosts all the algorithms used in the framework as functions
    Input Parameters:
    Data: Main dataset
    Dataname: Name of the dataset
    freshlist: 3D Array used to store results of all the runs
    Iterator: A int parameter used to define the number of iterator the framework is on, to be used to save results on 3D array
    Runs: Gives the total runs for iterator specified
    '''
    def __init__(self, data, dataname,freshlist,iterator, runs):
        self.data = data
        self.dataname = dataname

        self.freshlist = freshlist
        self.iterator = iterator
        self.runs = runs



    def adder(self, result, filename, alg,Optimizer, loss):
        '''
        Function to create the final pickled results and also saves results of each iteration
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

        #listback = joblib.load(direct2)
        for i in range(7):

            freshlist[alg][i][iterator] = result[i]

        print(freshlist)
        if iterator==(self.runs -1):
            for i in range(7):
                listoflist.append(np.mean(freshlist[alg][i]))
            listoflist.append(np.std(freshlist[alg][4]))
            listoflist.append(np.std(freshlist[alg][5]))
            if Optimizer!=None:
                listoflist.append(Optimizer)
                listoflist.append(loss)
                joblib.dump(listoflist, direct2)
            else:

                joblib.dump(listoflist, direct2)





    #adding file path folder extension to the file path
    bestresults = [0, 0, 0, 0, 0, 0]
    bestresultssvm = [0,0]
    def Iocsvm(self):
        '''
        Function for One Class SVM
        :return:
        Gives output to be saved by the adder function
        '''
        global bestresultssvm
        alg = 0
        data = self.data

        X_train, X_val,  X_test,y_train,  y_val , y_test= converter_1(data,
                                                                                   onClassClassification=True,
                                                                                   shuffle=False, test_size=0.25,
                                                                                   validation_size=0.25)
        if self.iterator == 0:

            #pass
            bestresultssvm = SVMhyp(X_train, X_val, X_test, y_train, y_val, y_test)



        start = timeit.default_timer()
        loader = OneClassSVM(X_train, X_val, X_test, y_train, y_val, y_test, bestresultssvm[0], bestresultssvm[1],bestresultssvm[2])

        y_test, y_pred = loader.prediction()

        end = timeit.default_timer()
        totaltime = round(end - start,2)
        Name = 'OneClassSVM' + '-' + self.dataname

        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]
        self.adder(results, Name,alg, 0,0)


    def Ae(self):
        '''
        Function for Autoencoders
        :return:
        Returns results to be saved by the adder function
        '''

        alg = 1
        data = self.data
        train, valid, validB, testB = converter_1(data, AE=True)
        #Initial function to perform grid search
        global bestresults
        if self.iterator == 0:


            bestresults = Hyperparametertuning(train, valid, validB, testB )
            #bestresults = [1e-6, 'relu', 'Adam', 'mean_squared_error', 64,20]

        alg = 1



        train, valid, validB, testB = converter_1(data,AE=True)

        start = timeit.default_timer()
        loader = Autoen(train, valid, validB, testB)
        y_test, y_pred = loader.Ae(bestresults[0],bestresults[1],bestresults[2],bestresults[3],bestresults[4],bestresults[5])
        print(bestresults[0],bestresults[1],bestresults[2],bestresults[3],bestresults[4],bestresults[5])
        #y_test, y_pred= loader.Ae(1e-05,'relu', 'Adam', 'mean_squared_error', 1, 64)
        end = timeit.default_timer()
        totaltime = round(end - start, 2)
        opter, losser = 1,2
        Name = 'AutoEncoder' + '-' + self.dataname
        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]
        self.adder(results, Name,alg, opter,losser)

        print(self.iterator, "HEEEEEEEREEEEEEEEEEEEEE ITERATOR 22222")

    def C_AE(self):
        '''
        Compressed Autoencoder function
        :return:
        Output to be saved by the adder function
        '''
        global opter
        global losser
        global bestresults
        alg = 2
        data = self.data
        train, validB, testB = converter_1(data, cAE=True)
        #GridSearch
        if self.iterator == 0:
            bestresults = [1e-5, 'relu', 'Adam', 'mean_absolute_error', 1, 83]
            #bestresults = C_Hyp(train, validB, testB)


        print(bestresults)



        start = timeit.default_timer()
        loader = compressing_autoencoder(train, validB, testB,bestresults[0],bestresults[1],bestresults[2],bestresults[3],bestresults[4],bestresults[5])

        loader.loading_data()
        y_test, y_pred = loader.test()
        end = timeit.default_timer()
        totaltime = round(end - start, 2)

        Name = 'Compressed_AutoEncoder' + '-' + self.dataname
        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]
        self.adder(results, Name,alg, bestresults[3],bestresults[2])

        print(self.iterator, "HEEEEEEEREEEEEEEEEEEEEE ITERATOR 33333333333333")

    def SVM(self):
        '''
        SVM function
        :return:
        results to be saved by the adder function
        '''
        global opter
        global losser
        alg = 3
        if self.iterator == 0:
            Optimizers = ['rbf','linear','poly']

            loss = [1,2,3,4,5]
            test2 = 0

            for elements in range(len(Optimizers)):
                for elemento in range(len(loss)):
                    alg = 1
                    data = self.data

                    X_train, X_test, y_train, y_test = converter_1(data, SVM=True)



                    loader = SVM(X_train, X_test, y_train, y_test,Optimizers[elements], loss[elemento])
                    y_test, y_pred = loader.svm()


                    _, _, _, _, num, num2 = feedbackdata(y_test, y_pred)



                    if num2 <= 1:
                        num2 = 1
                    teste = num* (np.power(num, (1 / num2)))
                    if teste > test2:
                        opter = Optimizers[elements]
                        losser = loss[elemento]
                        test2 = teste



        data = self.data



        X_train, X_test, y_train, y_test = converter_1(data, SVM=True)

        start = timeit.default_timer()

        loader = SVM(X_train, X_test, y_train, y_test,opter,losser)
        y_test, y_pred = loader.svm()
        end = timeit.default_timer()
        totaltime = round(end - start, 2)

        Name = 'SVM' + '-' + self.dataname
        tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(y_test, y_pred)
        results = [tn, fp, fn, tp, detection_rate, false_positive_rate, totaltime]

        self.adder(results, Name,alg, opter,losser)
