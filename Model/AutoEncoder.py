from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from measures.metrics import feedbackdata
from skopt import gp_minimize
from sklearn.preprocessing import StandardScaler


class Autoen:

    def __init__(self, train, valid, validB, testB):
        self.train = train
        self.valid = valid
        self.validB = validB
        self.testB = testB

    def Ae(self, learning_rate, activation_function, optimizer, loss, batch_size,hidden_dimensions):

        epochs = 50
        # learning_rate = 1e-5
        #  activation_function = 'relu'
        # num_dense_layers = 1
        # batch_size = 64
        # df = self.df

        '''
            Accepts single DF file with anomalous and normal data
            defined by the approved labels of the framework
            DF: Dataframe input for the algorithm to work on.
            Epochs: Int input for the number of epochs default size: 100
            Batch_size: Int input for the number of batch_size default size 32

            returns;
                1- Predicted Y and Real Y to outputresult
                2- Total time of the algorithm
                3- Graph showing the threshold for error distribtution and threshold (Optional) (removed)
            '''
        import tensorflow
        import matplotlib.pyplot as plt
        import seaborn as sns

        import pandas as pd
        import numpy as np
        from pylab import rcParams
        import tensorflow as tf
        from keras.models import Model, load_model
        from keras.layers import Input, Dense
        from keras.callbacks import ModelCheckpoint, TensorBoard
        from keras import regularizers

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, precision_recall_curve
        from sklearn.metrics import recall_score, classification_report, auc, roc_curve
        from sklearn.metrics import precision_recall_fscore_support, f1_score
        from numpy.random import seed
        import timeit
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)
        SEED = 123  # used to help randomly select the data points
        DATA_SPLIT_PCT = 0.2
        rcParams['figure.figsize'] = 8, 6
        LABELS = ["Normal", "Break"]
        fault = 'Class'
        colm = list(self.train.columns)

        for item in colm:
            if (item == ("FaultNumber" or "Class" or "class" or "faultnumber")):
                fault = item

        df_train_0_x = self.train.drop([fault], axis=1)
        df_test = self.testB
        df_valid = self.valid
        df_validB = self.validB
        print(self.validB.shape)
        # made changes here
        df_valid_0_x = self.valid.drop([fault], axis=1).values

        # Scalerizing the data
        scaler = StandardScaler().fit(df_train_0_x)
        df_train_0_x_rescaled = scaler.transform(df_train_0_x)
        # validation of tensorflow
        df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)

        # validation for error threshold
        df_valid_x_rescaled = scaler.transform(df_valid.drop([fault], axis=1))
        df_valid_xB = scaler.transform((df_validB.drop([fault], axis=1)))
        # testing
        # df_test_x_rescaled = scaler.transform(df_test.drop([fault], axis=1))

        nn_samples = df_valid[df_valid[fault] == 0].shape[0]
        na_samples = df_valid[df_valid[fault] == 1].shape[0]
        n_features = df_train_0_x_rescaled.shape[1]

        nb_epoch = epochs
        batch_size = batch_size
        input_dim = df_train_0_x_rescaled.shape[1]  # num of predictor variables,
        encoding_dim = input_dim * hidden_dimensions
        hidden_dim = int(encoding_dim / 2)
        learning_rate = learning_rate

        # Start of the Algorithm layer formation

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation=activation_function,
                        activity_regularizer=regularizers.l1(learning_rate))(
            input_layer)
        encoder = Dense(hidden_dim, activation=activation_function)(encoder)
        decoder = Dense(hidden_dim, activation=activation_function)(encoder)
        decoder = Dense(encoding_dim, activation=activation_function)(decoder)
        decoder = Dense(input_dim, activation="linear")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # to save model data on the drive, no using till testing

        active_idle = [1] * len(df_train_0_x_rescaled)

        history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled, epochs=nb_epoch,
                                  batch_size=batch_size, shuffle=True,
                                  validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                                  sample_weight=np.asarray(active_idle),
                                  verbose=1)
        # Measuring Time
        print(history.history['accuracy'][-1])
        # ERRROR threshold setting

        valid_x_predictions = autoencoder.predict(df_valid_xB)
        # FOR NORMAL

        debug = True

        mse = np.mean(np.power(df_valid_xB - valid_x_predictions, 2), axis=1)

        classes_normal = [0] * nn_samples
        classes_anomal = [1] * na_samples
        errors = mse
        print(len(errors), "Error shape")
        classes = df_validB[fault]
        print(len(classes), "Shape of CLass")
        n_perc_min = 0
        n_perc_max = 98

        best_threshold = 0
        precision_W_best = 0
        fscore_A_best = 0
        AUC_best = 0
        fscore_N_best = 0
        recall_A_best = 0
        recall_N_best = 0
        recall_W_best = 0
        fscore_W_best = 0
        perc_best = 0
        fps = []
        fns = []
        tps = []
        tns = []
        n_percs = []
        precs = []
        recalls = []
        fscores = []

        # Looping for error threshold calculation between 0 to 100 percentile

        for n_perc in np.linspace(n_perc_min, n_perc_max + 2, 1000):
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
            print(predictions.count(0))
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
            print(precision_N, recall_N, fscore_N)
            print(precision_A, recall_A, fscore_A)
            print( precision_W, recall_W, fscore_W)
            print(precision_W, recall_W, fscore_W)
            tn, fp, fn, tp, detection_rate, false_positive_rate = feedbackdata(classes, predictions)

            false_positive_rate = 1 / false_positive_rate
            false_positive_rate = false_positive_rate * 100

            #sq = np.square(1 - false_positive_rate)
            #sq2 = np.square(1 - detection_rate)
            perc = np.sqrt(np.square(1 - detection_rate) + np.square(false_positive_rate))
            # if fscore_A> recall_A_best:
            if fscore_W> fscore_W_best:
                precision_W_best = precision_W
                precision_N_best = precision_N
                precision_A_best = precision_A
                recall_W_best = recall_W
                recall_N_best = recall_N
                recall_A_best = recall_A
                fscore_W_best = fscore_W
                fscore_N_best = fscore_N
                fscore_A_best = fscore_A
                perc_best = perc
                tp_best = tp
                best_threshold = n_perc
        pax = np.percentile(np.asarray(errors), best_threshold)

        if debug:
            pass

        threshold_fixed = pax

        df_test = df_test
        df_test_x_rescaled = scaler.transform(df_test.drop([fault], axis=1))

        test_x_predictions = autoencoder.predict(df_test_x_rescaled)

        mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
        error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                                      'True_class': df_test[fault]})
        error_df_test = error_df_test.reset_index()
        # threshold_fixed = pax

        # plt.title("Reconstruction error for different classes")
        # plt.ylabel("Reconstruction error")
        # plt.xlabel("Data point index")
        # plt.show();
        pred_y = [1 if e > threshold_fixed else 0 for e in error_df_test.Reconstruction_error.values]

        return (error_df_test.True_class, pred_y)


bestresultz = []
best_accuracy = 0.0


def Hyperparametertuning(train, valid, validB, testB):
    import skopt
    from skopt import gp_minimize
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Categorical, Integer
    from skopt.plots import plot_convergence
    from skopt.plots import plot_objective, plot_evaluations
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import InputLayer, Input
    from tensorflow.python.keras.layers import Reshape, MaxPooling2D
    from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
    from tensorflow.python.keras.callbacks import TensorBoard
    from tensorflow.python.keras.optimizers import Adam
    from tensorflow.python.keras.models import load_model
    from skopt.utils import use_named_args
    import numpy as np

    path_best_model = '19_best_model.keras'
    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                             name='learning_rate')
    dim_activation = Categorical(categories=['relu', 'sigmoid'],
                                 name='activation_function')
    optimizer = Categorical(categories=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'SGD'], name='optimizer')
    loss = Categorical(
        categories=['binary_crossentropy', 'categorical_crossentropy', 'categorical_hinge', 'mean_absolute_error',
                    'mean_absolute_percentage_error', 'mean_squared_error', 'mean_squared_logarithmic_error'],
        name='loss')

    #dim_num_dense_layers = Integer(low=0, high=5, name='num_dense_layers')
    # batch_size = Integer(low=32, high=128, name='batch_size')
    batch_size = Categorical(categories=['32', '64', '128'], name='batch_size')
    hidden_dimensions = Integer(low=2, high=20, name='hidden_dimensions')
    dimensions = [dim_learning_rate,
                  dim_activation,
                  optimizer,
                  loss,
                  batch_size,
                  hidden_dimensions
                  ]

    @use_named_args(dimensions=dimensions)
    def Ae(learning_rate, activation_function, optimizer, loss, batch_size, hidden_dimensions):

        epochs = 50
        # learning_rate =1e-5
        # activation_function= 'relu'
        # num_dense_layers = 1
        # batch_size = 64
        batch_size = int(batch_size)
        # global best_accuracy
        # global bestresultz
        '''
            Accepts single DF file with anomalous and normal data
            defined by the approved labels of the framework
            DF: Dataframe input for the algorithm to work on.
            Epochs: Int input for the number of epochs default size: 100
            Batch_size: Int input for the number of batch_size default size 32

            returns;
                1- Predicted Y and Real Y to outputresult
                2- Total time of the algorithm
                3- Graph showing the threshold for error distribtution and threshold (Optional)
            '''

        import numpy as np
        from pylab import rcParams
        from keras.models import Model
        from keras.layers import Input, Dense
        from keras import regularizers
        from sklearn.preprocessing import StandardScaler
        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)
        SEED = 123  # used to help randomly select the data points
        DATA_SPLIT_PCT = 0.2
        rcParams['figure.figsize'] = 8, 6
        LABELS = ["Normal", "Break"]
        fault = 'Class'
        colm = list(train.columns)

        for item in colm:
            if (item == ("FaultNumber" or "Class" or "class" or "faultnumber")):
                fault = item

        df_train_0_x = train.drop([fault], axis=1)
        df_test = testB
        df_valid = valid
        df_validB = validB
        print(validB.shape)
        # made changes here
        df_valid_0_x = valid.drop([fault], axis=1).values

        # Scalerizing the data
        scaler = StandardScaler().fit(df_train_0_x)
        df_train_0_x_rescaled = scaler.transform(df_train_0_x)
        # validation of tensorflow
        df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)

        # validation for error threshold
        df_valid_x_rescaled = scaler.transform(df_valid.drop([fault], axis=1))
        df_valid_xB = scaler.transform((df_validB.drop([fault], axis=1)))
        # testing
        # df_test_x_rescaled = scaler.transform(df_test.drop([fault], axis=1))

        nn_samples = df_valid[df_valid[fault] == 0].shape[0]
        na_samples = df_valid[df_valid[fault] == 1].shape[0]
        n_features = df_train_0_x_rescaled.shape[1]

        nb_epoch = epochs
        batch_size = batch_size
        input_dim = df_train_0_x_rescaled.shape[1]  # num of predictor variables,
        encoding_dim = input_dim * hidden_dimensions
        hidden_dim = int(encoding_dim / 2)
        learning_rate = learning_rate

        # Start of the Algorithm layer formation
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation=activation_function,
                        activity_regularizer=regularizers.l1(learning_rate))(
            input_layer)
        encoder = Dense(hidden_dim, activation=activation_function)(encoder)
        decoder = Dense(hidden_dim, activation=activation_function)(encoder)
        decoder = Dense(encoding_dim, activation=activation_function)(decoder)
        decoder = Dense(input_dim, activation="linear")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # to save model data on the drive, no using till testing

        active_idle = [1] * len(df_train_0_x_rescaled)

        history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled, epochs=nb_epoch,
                                  batch_size=batch_size, shuffle=True,
                                  validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                                  sample_weight=np.asarray(active_idle),
                                  verbose=1)
        # Measuring Time

        # ERRROR threshold setting

        valid_x_predictions = autoencoder.predict(df_valid_xB)
        # FOR NORMAL

        debug = True

        mse = np.mean(np.power(df_valid_xB - valid_x_predictions, 2), axis=1)

        classes_normal = [0] * nn_samples
        classes_anomal = [1] * na_samples
        errors = mse
        print(len(errors), "Error shape")
        classes = df_validB[fault]

        n_perc_min = 0
        n_perc_max = 99

        best_threshold = n_perc_max
        precision_W_best = 0
        recall_N_best = 0
        fscore_A_best = 0
        fscore_N_best = 0
        recall_A_best = 0
        precision_N_best = 0
        fscore_W_best = 0
        fps = []
        fns = []
        tps = []
        tns = []
        n_percs = []
        precs = []
        recalls = []
        fscores = []
        fscore_A = 0
        recall_N = 0
        recall_A = 0
        # Looping for error threshold calculation between 0 to 100 percentile

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
            print(len(classes), "Shape of CLass")
            print(predictions.count(0))
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

            # if precision_W> precision_W_A_best:
            if fscore_W> fscore_W_best:
                precision_W_best = precision_W
                precision_N_best = precision_N
                precision_A_best = precision_A
                recall_W_best = recall_W
                recall_N_best = recall_N
                recall_A_best = recall_A
                fscore_W_best = fscore_W
                fscore_N_best = fscore_N
                fscore_A_best = fscore_A
                tp_best = tp
                best_threshold = n_perc
        pax = np.percentile(np.asarray(errors), best_threshold)

        if debug:
            pass

        threshold_fixed = pax
        global best_accuracy
        global path_best_model
        # If the classification accuracy of the saved model is improved ...
        if fscore_W_best > best_accuracy:
            # Save the new model to harddisk.
            # autoencoder.save(path_best_model)learning_rate,  activation_function,  optimizer, loss ,num_dense_layers, batch_size
            bestresultz.append(learning_rate)
            bestresultz.append(activation_function)
            bestresultz.append(optimizer)
            bestresultz.append(loss)
            # bestresultz.append(num_dense_layers)

            bestresultz.append(batch_size)
            bestresultz.append(hidden_dimensions)
            # Update the classification accuracy.
            best_accuracy = fscore_W_best

        # Delete the Keras model with these hyper-parameters from memory.
        del autoencoder

        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()

        # NOTE: Scikit-optimize does minimization so it tries to
        # find a set of hyper-parameters with the LOWEST fitness-value.
        # Because we are interested in the HIGHEST classification
        # accuracy, we need to negate this number so it can be minimized.

        # print(recall_A, 'HERE BC')
        return -fscore_W_best

    import pandas as pd

    default_parameters = [1e-5, 'relu', 'Adam', 'mean_squared_error', '64',2]
    # default_parameters = ['Adam', 'mean_squared_error']
    search_result = gp_minimize(func=Ae, dimensions=dimensions, acq_func='EI', n_calls=30, x0=default_parameters)

    print(bestresultz)
    # print(best_accuracy)
    return bestresultz[-6:]
