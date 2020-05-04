import pandas as pd
from pandas import Series
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from keras import regularizers
from glob import glob
from sklearn.preprocessing import LabelEncoder
from keras.utils import plot_model
from termcolor import colored
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class compressing_autoencoder:
    """
    Desctiption:
    ------------
        A compressed auto-encoder model for timeseries data.
        
    Parameters
    ----------
    trainDir : string, (required)
        The directory where the training datasets as csv files is, the training data should only contain normal behavior without anomalies.
        ex: data/training (without backslash at the end)
        
    valDir: string, (required)
        The directory where the validationg data is, it is used to calculate the error threshold, the validation data should contain normal and anomalous data.
        ex: data/testing (without backslash at the end)

    dataDelimiter: string, optional(default=';')
        The delimiter used in the training data csv files.
    
    dataDecimal: string, optional(default=',')
        The decimal symbol used for the values in the training data csv file.
        
    sequenceLength: int, optional(default=100)
        The sequence length that the model processes for each epoch.
        
    modelBatchSize: int, optional(default=64)
        The model batch size.
    
    modelEpochs: int, optional(default=20)
        The model training epochs.
        
    showModelSummary: boolean, optional(default=False)
        Showing the summary of the model created by the instance.
        
    saveModelStructureToDisk: boolean, optional(default=False)
        Saving the model overview to a jpg file in the working directory.       
        
        
    Returns
    -------
        After creating an instance of this class, the model can be accesses by the class function model()
        Ex:
        autoEncoder = compressing_autoencoder(trainingPath,testingPath)
        model = autoEncoder.model()
        
        example using the model to reconstruct a sequence: 
        reconstructedSequence = model.predict(inputSequence)
    """

    def __init__(self, trainDir, valDir, testDir, learning_rate, activation_function, optimizer, loss, num_dense_layers, batch_size,
                 sequenceLength=200,
                 modelBatchSize=64, modelEpochs=20, showModelSummary=False, saveModelStructureToDisk=False,
                 threshold_fn_percentage=.0005, valDirfault=None, testfault=None):
        print("initializing ..")
        self.successMsg = '\x1b[0;30;42m' + 'Success!' + '\x1b[0m'
        # data params
        self.threshold_fn_percentage = threshold_fn_percentage
        self.trainDir = trainDir
        self.testDir = testDir
        self.valDir = valDir

        self.sequenceLength = sequenceLength
        # model params
        self.valdirfault = valDirfault
        self.modelBatchSize = batch_size
        self.testfault = testfault
        self.modelEpochs = modelEpochs
        self.showModelSummary = showModelSummary
        self.saveModelStructureToDisk = saveModelStructureToDisk
        self.Optimizer = optimizer
        self.Loss = loss
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.num_dense_layers = num_dense_layers
         # self.call_libraries()
        # self.loading_data()

    timetotal = 0

    def loading_data(self):
        """
        Desctiption:
        ------------
            This function loads the csv files to Pandas dataframes, and concatenates the csv training data files into one dataframe.
        """
        print("Starting here")
        df = self.trainDir
        fault = 'Class'
        col = df.columns
        for elements in col:
            if "FaultNumber" or "Class" or "class" or "faultnumber" in elements:
                fault = elements
        # df = df.loc[df[fault]==0]
        df = df.drop([fault], axis=1)
        self.trainingData = pd.DataFrame(df)
        print("loading_data")
        self.preprocessing_Data()

    def preprocessing_Data(self):
        """
        Desctiption:
        ------------
            This function converts the dataframe into 3d numpy arary, the output numpyArray is used as input to the lstm model.
        """

        print("preprocessing data ..")
        trainingData = self.trainingData

        def myround(x, base=self.sequenceLength):
            a = base * np.round(x / base)
            if a > x:
                return a - self.sequenceLength
            else:
                return a
            # return base*np.round(x/base)

        valx = myround(self.trainingData.shape[0])
        valx = int(valx)
        print(valx, "heeee")
        trainingData = trainingData.iloc[0:valx, :]

        trainDataColumns = trainingData.columns
        print(trainingData.shape, "HEREE")
        # setting up number of sequences and number of columns
        self.numOfSequences = int(len(trainingData.index) / self.sequenceLength)
        self.numOfColumns = len(trainingData.columns)

        # scaling the data & saving the scaler for later use
        scaler = MinMaxScaler()
        scaled_trainingData = scaler.fit_transform(trainingData)
        # scaler_filename = "scaler.save"
        # joblib.dump(scaler, scaler_filename)
        self.scaler = scaler
        dfval = self.valDir
        col = dfval.columns
        for elements in col:
            if "FaultNumber" or "Class" or "class" or "faultnumber" in elements:
                fault = elements
                # for validation
        valDir = dfval
        self.valdirfault = valDir[fault]
        valDir = valDir.drop([fault], axis=1)
        valval = myround(valDir.shape[0])
        valval = int(valval)
        valDir = valDir.iloc[:valval, :]

        valDir = scaler.fit_transform(valDir)
        self.valDir = self.data_3d_reshape(valDir)

        # for test
        self.testfault = self.testDir[fault]
        self.testDir = self.testDir.drop([fault], axis=1)
        valtest = myround(self.testDir.shape[0])
        valtest = int(valtest)
        self.testDir = self.testDir.iloc[:valtest, :]
        self.testfault = self.testfault.iloc[:valtest]
        self.testDir = scaler.fit_transform(self.testDir)
        self.testDir = self.data_3d_reshape(self.testDir)

        # converting the scaled data to dataFrame

        trainingData = self.data_3d_reshape(scaled_trainingData)
        self.scaledTrainingData = trainingData
        print(self.successMsg)
        self.autoencoder_model(trainingData)
        print("pre processing data")

    def data_3d_reshape(self, Data):
        """
            Desctiption:
            ------------
                reshaping function, it takes as input numpyArray data and reshape it to 3d, the output is a 3d shape numpyArray which is the input to the LSTM model
            """
        print("reshaping ..")

        print(Data.shape, "DATA_3D_Reshaping")
        sequenceLength = self.sequenceLength
        numOfSequences = int(Data.shape[0] / sequenceLength)
        numOfColumns = int(Data.shape[1])
        print("sequenceLength: ", sequenceLength)
        print("numOfSequences: ", numOfSequences)
        print("numOfColumns: ", numOfColumns)
        boot = Data.shape[0]
        bootr = sequenceLength * numOfSequences
        print(bootr)

        final = Data[0:bootr, :]
        print(final.shape, 'shape of final ')
        return final.reshape(numOfSequences, sequenceLength, numOfColumns)

        print("dta_3d_reshape")

    def autoencoder_model(self, trainingData):
        """
        Desctiption:
        ------------
            This function creates the lstm autoencoder model with the given class parameters settings.
        """

        sequenceLength = self.sequenceLength
        numOfColumns = self.numOfColumns

        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(
            LSTM(sequenceLength, activation=self.activation_function, input_shape=(None, numOfColumns), return_sequences=True))
        lstm_autoencoder.add(LSTM(120, activation=self.activation_function, return_sequences=True))
        lstm_autoencoder.add(LSTM(120, activation=self.activation_function, return_sequences=True))
        lstm_autoencoder.add(LSTM(60, activation=self.activation_function))
        lstm_autoencoder.add(RepeatVector(sequenceLength))
        # Decoder
        lstm_autoencoder.add(LSTM(sequenceLength, activation=self.activation_function, return_sequences=True))
        lstm_autoencoder.add(LSTM(120, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(120, activation='relu', return_sequences=True))
        lstm_autoencoder.add(TimeDistributed(Dense(numOfColumns)))
        print(self.successMsg)
        self.fitting_model(lstm_autoencoder, trainingData)
        print("autoencoder modle")

    def fitting_model(self, lstm_autoencoder, trainingData):
        Optimizer = self.Optimizer
        Loss = self.Loss
        """
        Desctiption:
        ------------
            This function fits the model created by the function autoencoder_model() to the training data given as input
        """

        batchSize = self.modelBatchSize
        modelEpochs = self.modelEpochs
        if (self.showModelSummary == True):
            print("model summary:")
            print(lstm_autoencoder.summary())

        import timeit

        lstm_autoencoder.compile(optimizer=Optimizer, loss=Loss, metrics=['acc'])
        history = lstm_autoencoder.fit(trainingData, trainingData, epochs=modelEpochs, verbose=1, batch_size=batchSize)

        self.model = lstm_autoencoder

        self.threshold = self.calculate_error_threshold()
        global timetotal

        self.model = lstm_autoencoder
        self.load_testData()
        print("fitting model")

    def model(self):
        """
         Desctiption:
        ------------
            This function returns the final autoencoder model, the model can be used to reconstruct the input sequence. 
        """
        print("model")
        return self.model

    def calculate_error_threshold(self):
        '''
        def myround(x, base=self.sequenceLength):
            a = base * np.round(x / base)
            if a > x:
                return a - 50
            else:
                return a
            # return base*np.round(x/base)

        valx = myround(self.valdirfault.shape[0])
        valx = int(valx)
        print(valx, "heeee")
        '''
        valx = self.valDir.shape[0] * self.valDir.shape[1]
        self.valdirfault = self.valdirfault.iloc[0:valx]

        predictions = self.model.predict(self.valDir)
        column = predictions.shape[2]

        predictions = predictions.reshape(valx, column)
        self.valDir = self.valDir.reshape(valx, column)

        print(predictions)
        # validationData = self.scaledTrainingData.reshape(212900, 16)
        # predictions = predictions.reshape(212900, 16)
        mse = np.mean(np.power(self.valDir - predictions, 2), axis=1)
        print(mse)
        print(mse.shape)
        debug = True
        errors = mse
        print(len(errors), "Error shape")

        classes = self.valdirfault
        print(len(classes), "Shape of CLass")
        n_perc_min = 0
        n_perc_max = 98

        best_threshold = n_perc_max
        fscore_A_best = 0
        fscore_N_best = 0
        recall_N_best = 0
        fps = []
        fns = []
        tps = []
        tns = []
        n_percs = []
        precs = []
        recalls = []
        fscores = []

        for n_perc in range(n_perc_min, n_perc_max + 2):
            error_threshold = np.percentile(np.asarray(errors), n_perc)
            if debug:
                print("Try with percentile: %s (threshold: %s)" % (
                    n_perc, error_threshold))

            predictions = []
            for e in errors:
                if e > error_threshold:
                    predictions.append(1)
                else:
                    predictions.append(0)
            print(len(predictions), "Prediction shape")
            precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
                classes, predictions, average='binary', pos_label=0)
            precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
                classes, predictions, average='binary', pos_label=1)
            precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
                classes, predictions, average='weighted')

            tn, fp, fn, tp = confusion_matrix(classes, predictions).ravel()
            fscores.append(fscore_W)
            precs.append(precision_W)
            recalls.append(recall_W)
            n_percs.append(n_perc)
            fps.append(fp)
            fns.append(fn)
            tps.append(tp)
            tns.append(tn)

            if fscore_A> fscore_N_best:
                precision_W_best = precision_W
                precision_N_best = precision_N
                precision_A_best = precision_A
                recall_W_best = recall_W
                recall_N_best = recall_N
                recall_A_best = recall_A
                fscore_W_best = fscore_W
                fscore_N_best = fscore_N
                fscore_A_best = fscore_A
                best_threshold = n_perc
        pax = np.percentile(np.asarray(errors), best_threshold)

        '''
        self.mse = mse
        error_df = pd.DataFrame({'reconstruction_error': mse})
        error_df['groundTruth'] = 0

        reconstructionErrors = error_df.reconstruction_error.values
        truthies = error_df.groundTruth.values

        # calculating threshold:
        threshold = 0
        acceptable_n_FN = self.threshold_fn_percentage * 212900
        print("acceptable_n_FN: ", acceptable_n_FN)
        FN = 0
        TP = 0
        while (threshold <= 1):

            threshold += .005
            y_pred = [1 if e > threshold else 0 for e in reconstructionErrors]
            yPred = np.array(y_pred)
            conf_matrix = confusion_matrix(error_df.groundTruth, y_pred, labels=[0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()
            FN = fp
            TP = tn

            if FN < acceptable_n_FN:
                break
        self.threshold = threshold
        print("calculate threshold")
        '''
        threshold = pax

        # self.threshold = threshold
        return threshold

    def adjusting_n_rows(self, df):
        print(type(df))
        while (df.shape[0] % self.sequenceLength != 0):
            df.drop(df.tail(1).index, inplace=True)

        print("adjusting n rows")
        return df

    def load_testData(self):
        pass
        # df = self.testDir
        # to find the testdata and error finding
        # col = df.columns
        # for elements in col:
        # if "FaultNumber" or "Class" or "class" or "faultnumber" in elements:
        # fault = elements
        # df = df.drop([fault], axis=1)

        # testData = self.scaler.transform(df)
        # self.testData = testData
        # print("load testdata")

    def test(self):
        global timetotal
        testData = self.testDir
        threshold = self.threshold
        predictions = self.model.predict(testData)

        rowz = predictions.shape[0] * predictions.shape[1]
        print("ROWZ", rowz)
        column = predictions.shape[2]
        print("predictions shape : ")
        print(predictions.shape)
        predictions = predictions.reshape(rowz, column)
        testData = testData.reshape(rowz, column)
        # self.post_predictions = predictions
        # self.post_testData = testData

        mse = np.mean(np.power(testData - predictions, 2), axis=1)
        error_df = pd.DataFrame({'reconstruction_error': mse})

        print(mse.max())
        print(mse.min())
        print(threshold, "Threshold")
        # self.error_df = error_df
        reconstructionErrors = error_df.reconstruction_error.values
        # self.testReconstructionErrors = reconstructionErrors
        y_pred = []
        counter = 0
        for e in mse:
            if e > threshold:
                y_pred.append(1)
            if e <= threshold:
                y_pred.append(0)
        # y_pred = [1 if e > threshold else 0 for e in mse]
        # self.anomalyPredictions = y_pred
        # print(self.anomalyPredictions)
        print(y_pred)

        self.testfault = self.testfault[0:rowz]

        return self.testfault.values.ravel(), y_pred


best_accuracyC = 0
bestresultC = []


def C_Hyp(trainDir2, valDir2, testDir2, modelEpochs=20):
    import skopt
    from skopt import gp_minimize
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Categorical, Integer
    from skopt.plots import plot_convergence
    from skopt.plots import plot_objective, plot_evaluations
    from tensorflow.python.keras import backend as K
    from keras.models import Model, load_model, Sequential
    from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
    from skopt.utils import use_named_args
    import numpy as np

    path_best_model = '19_best_model.keras'
    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                             name='learning_rate')
    dim_activation = Categorical(categories=['relu', 'softmax'],
                                 name='activation_function')
    optimizer = Categorical(categories=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'SGD'], name='optimizer')
    loss = Categorical(
        categories=['binary_crossentropy', 'categorical_crossentropy', 'categorical_hinge', 'mean_absolute_error',
                    'mean_absolute_percentage_error', 'mean_squared_error', 'mean_squared_logarithmic_error'],
        name='loss')
    dim_num_dense_layers = Integer(low=0, high=1, name='num_dense_layers')
    batch_size = Integer(low=32, high=128, name='batch_size')

    dimensions = [dim_learning_rate,
                  dim_activation,
                  optimizer,
                  loss,
                  dim_num_dense_layers,
                  batch_size]

    trainDir2 = pd.DataFrame(trainDir2)
    valDir2 = pd.DataFrame(valDir2)

    @use_named_args(dimensions=dimensions)
    def C_hyper(learning_rate, activation_function, optimizer, loss, num_dense_layers, batch_size):
        modelEpochs = 20
        valDir = valDir2
        sequenceLength2 = 100
        trainDir = trainDir2
        df = trainDir
        fault = 'Class'
        col = df.columns
        for elements in col:
            if "FaultNumber" or "Class" or "class" or "faultnumber" in elements:
                fault = elements
        # df = df.loc[df[fault]==0]
        df = df.drop([fault], axis=1)
        trainingData = pd.DataFrame(df)
        print("loading_data")

        #global numOfColumns
        # global valDir
        testDir = testDir2

        # global sequenceLength
        def data_3d_reshape(Data):

            print(Data.shape, "DATA_3D_Reshaping")
            sequenceLength = sequenceLength2
            numOfSequences = int(Data.shape[0] / sequenceLength)
            numOfColumns = int(Data.shape[1])
            print("sequenceLength: ", sequenceLength)
            print("numOfSequences: ", numOfSequences)
            print("numOfColumns: ", numOfColumns)
            boot = Data.shape[0]
            bootr = sequenceLength * numOfSequences
            print(bootr)

            final = Data[0:bootr, :]
            print(final.shape, 'shape of final ')
            return final.reshape(numOfSequences, sequenceLength, numOfColumns)

        def myround(x, base):
            a = base * np.round(x / base)
            if a > x:
                return a - 50
            else:
                return a
            # return base*np.round(x/base)

        valx = myround(trainingData.shape[0], sequenceLength2)
        valx = int(valx)
        print(valx, "heeee")
        trainingData = trainingData.iloc[0:valx, :]

        trainDataColumns = trainingData.columns
        print(trainingData.shape, "HEREE")
        # setting up number of sequences and number of columns
        numOfSequences = int(len(trainingData.index) / sequenceLength2)
        numOfColumns = len(trainingData.columns)

        # scaling the data & saving the scaler for later use
        scaler = MinMaxScaler()
        scaled_trainingData = scaler.fit_transform(trainingData)
        # scaler_filename = "scaler.save"
        # joblib.dump(scaler, scaler_filename)
        scaler = scaler
        dfval = valDir
        col = dfval.columns

        valDir = dfval
        valdirfault = valDir[fault]
        valDir = valDir.drop([fault], axis=1)
        valval = myround(valDir.shape[0], sequenceLength2)
        valval = int(valval)
        valDir = valDir.iloc[:valval, :]

        valDir = scaler.fit_transform(valDir)
        alDir = data_3d_reshape(valDir)

        # for test
        testfault = testDir[fault]
        testDir = testDir.drop([fault], axis=1)
        valtest = myround(testDir.shape[0], sequenceLength2)
        valtest = int(valtest)
        testDir = testDir.iloc[:valtest, :]
        testfault = testfault.iloc[:valtest]
        testDir = scaler.fit_transform(testDir)
        testDir = data_3d_reshape(testDir)

        # converting the scaled data to dataFrame

        trainingData = data_3d_reshape(scaled_trainingData)
        scaledTrainingData = trainingData

        sequenceLength = sequenceLength2
        numOfColumns = numOfColumns

        def autoencoder_model(trainingData):
            """
            Desctiption:
            ------------
                This function creates the lstm autoencoder model with the given class parameters settings.
            """

            lstm_autoencoder = Sequential()
            # Encoder
            lstm_autoencoder.add(
                LSTM(sequenceLength, activation=activation_function, input_shape=(None, numOfColumns), return_sequences=True))
            lstm_autoencoder.add(LSTM(120, activation=activation_function, return_sequences=True))
            lstm_autoencoder.add(LSTM(120, activation=activation_function, return_sequences=True))
            lstm_autoencoder.add(LSTM(60, activation=activation_function))
            lstm_autoencoder.add(RepeatVector(sequenceLength))
            # Decoder
            lstm_autoencoder.add(LSTM(sequenceLength, activation=activation_function, return_sequences=True))
            lstm_autoencoder.add(LSTM(120, activation='relu', return_sequences=True))
            lstm_autoencoder.add(LSTM(120, activation='relu', return_sequences=True))
            lstm_autoencoder.add(TimeDistributed(Dense(numOfColumns)))


            # global modelEpochs



            lstm_autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            history = lstm_autoencoder.fit(trainingData, trainingData, epochs=modelEpochs, verbose=1,
                                           batch_size=batch_size)
            history_dict = history.history
            accuracy = history.history['accuracy'][-1]
            return accuracy
        accuracy = autoencoder_model(trainingData)


        global best_accuracyC
        global path_best_model
        # If the classification accuracy of the saved model is improved ...
        if accuracy > best_accuracyC:
            # Save the new model to harddisk.
            # autoencoder.save(path_best_model)learning_rate,  activation_function,  optimizer, loss ,num_dense_layers, batch_size
            bestresultC.append(learning_rate)
            bestresultC.append(activation_function)
            bestresultC.append(optimizer)
            bestresultC.append(loss)
            bestresultC.append(num_dense_layers)

            bestresultC.append(batch_size)

            # Update the classification accuracy.
            best_accuracyC = accuracy

        # Delete the Keras model with these hyper-parameters from memory.
        #del lstm_autoencoder

        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()

        # NOTE: Scikit-optimize does minimization so it tries to
        # find a set of hyper-parameters with the LOWEST fitness-value.
        # Because we are interested in the HIGHEST classification
        # accuracy, we need to negate this number so it can be minimized.
        return -accuracy

    default_parameters = [1e-5, 'relu', 'Adam', 'mean_absolute_error', 1, 83]
    search_result = gp_minimize(func=C_hyper, dimensions=dimensions, acq_func='EI', n_calls=50, x0=default_parameters)

    print(bestresultC)
    print(best_accuracyC)
    return bestresultC[-6:]
