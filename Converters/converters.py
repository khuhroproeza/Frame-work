def converter_1(data, test_size=0.25, unsupervised=False, label=False, validation_size=None,
                onClassClassification=False, AE=False,
                random_state=None, shuffle=True, cAE=False, SVM=False, stratify=None, ):
    import pandas as pd
    import numpy as np
    '''This function will return train, test and validation based on the percentage defined on the parameter
        Parameters
       -------------------
       X : Dataframe_like
       Requires a dataframe having normal(FaultFree) data items with column names

       Y: Dataframe_like
       Requires a dataframe having anomaly(Faulty) data with column name

       test_size: float, int or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``

        unsupervised: Boolean, optional (Default=False)
        This parameter takes in the boolean input
        If True the converter converts the data into unsupervised dataframe
        combining both faulty and fault free data.

        Label: Boolean, optional( Default=False)
        This function takes in the boolean input to define the label parameter
        and works with the unsupervised parameter.
        If true it also returns a a lebel from the unsupervised dataframe as a
        seperate array. Its optional and should be used to find the accuracy of the
        algorithm.

        validation_size : float, int or None, optional (default= None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the train to include in the validation split. If int, represents the
        absolute number of train samples.


        random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

        shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

        stratify : array-like or None (default=None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.

        Returns
        -------
        splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs. If validation is defined then
        it will return X_train, X_test, X_validation, y_train, y_test, y_validation

    '''
    # Extract the name of class or fault number or label etc
    fault = 'Class'
    colm = list(data.columns)

    for item in colm:

        if item == ("FaultNumber" or "Class" or "class" or "faultnumber"):
            fault = item
    # Separate data and label for normal
    X = data[data[fault] == 0]
    Y = data[data[fault] == 1]

    x_label = X[fault].values
    X_data = X.drop(fault, axis=1).values

    # Separate data and label for anomaly
    y_label = Y[fault].values
    Y_data = Y.drop(fault, axis=1).values
    # Import splitting libraries
    # Now go for splitting the data based on classification approach e.g binary or onClassClassification
    from sklearn.model_selection import train_test_split

    if (onClassClassification != False and validation_size != None):

        faulty = (data[data[fault] == 1])
        faultfree = (data[data[fault] == 0])

        # faultfree = sampling(faultfree)
        dshape = faultfree.shape[0]
        dshape = 0.50 * dshape
        dshape = int(dshape)
        dshape1 = int(0.20 * dshape)
        dshape2 = dshape + dshape1
        x1 = faultfree.iloc[:dshape, :]
        x2 = faultfree.iloc[dshape:dshape2, :]
        x3 = faultfree.iloc[dshape2:, :]
        test = (x3, faulty)
        faultys = faulty.shape[0]
        fshape = int(0.50 * faultys)
        test = faulty.iloc[:fshape, :]
        valid = faulty.iloc[fshape:, :]
        testl = [x3, test]
        testB = pd.concat(testl, ignore_index=True)
        validl = [x2, valid]
        validB = pd.concat(validl, ignore_index=True)
        train = x1
        valid = x2

        y_train = train[fault].values
        X_train = train.drop(fault, axis=1).values

        y_val_n = validB[fault].values
        X_val_n = validB.drop(fault, axis=1).values

        y_test = testB[fault].values
        X_test = testB.drop(fault, axis=1).values

        return X_train, X_val_n, X_test, y_train, y_val_n, y_test


    elif (SVM != False):

        df = data.drop(fault, axis=1)
        dft = data[fault]

        X1 = (df)
        y1 = (dft)

        X_train, X_test, y_train, y_test = train_test_split(X1, y1, random_state=0)

        return X_train, X_test, y_train, y_test

    elif (AE != False):

        faulty = (data[data[fault] == 1])
        faultfree = (data[data[fault] == 0])

        # faultfree = sampling(faultfree)
        dshape = faultfree.shape[0]
        dshape = 0.50 * dshape
        dshape = int(dshape)
        dshape1 = int(0.20 * dshape)
        dshape2 = dshape + dshape1
        x1 = faultfree.iloc[:dshape, :]
        x2 = faultfree.iloc[dshape:dshape2, :]
        x3 = faultfree.iloc[dshape2:, :]
        test = (x3, faulty)
        faultys = faulty.shape[0]
        fshape = int(0.50 * faultys)
        test = faulty.iloc[:fshape, :]
        valid = faulty.iloc[fshape:, :]
        testl = [x3, test]
        testB = pd.concat(testl, ignore_index=True)
        validl = [x2, valid]
        validB = pd.concat(validl, ignore_index=True)
        train = x1
        valid = x2
        return train, valid, validB, testB

    elif (cAE != False):
        faulty = (data[data[fault] == 1])
        faultfree = (data[data[fault] == 0])

        # faultfree = sampling(faultfree)
        dshape = faultfree.shape[0]
        dshape = 0.50 * dshape
        dshape = int(dshape)
        dshape1 = int(0.20 * dshape)
        dshape2 = dshape + dshape1
        x1 = faultfree.iloc[:dshape, :]
        x2 = faultfree.iloc[dshape:dshape2, :]
        x3 = faultfree.iloc[dshape2:, :]
        test = (x3, faulty)
        faultys = faulty.shape[0]
        fshape = int(0.50 * faultys)
        test = faulty.iloc[:fshape, :]
        valid = faulty.iloc[fshape:, :]
        testl = [x3, test]
        testB = pd.concat(testl, ignore_index=True)
        validl = [x2, valid]
        validB = pd.concat(validl, ignore_index=True)
        train = x1
        valid = x2
        return train, validB, testB
