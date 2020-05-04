class SVM:
    def __init__(self, X_train, X_test, y_train, y_test, kernel, C):
        """
        SVM Algorithm
        df: Dataframe with labeled dataset
        Cname: Name of the CLassifier
        Dname: Name of the Dataset

        Returns:
            1. Predicted Y and actual Y
            2. Total time of the algorithm training

        :type train: if True Model trains again, if false : Model picks up last trained weights
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.kernel = kernel
        self.C = C





    def svm(self):

        from sklearn import datasets
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.externals import joblib

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        kernel = self.kernel
        C = self.C


        #If no weight file in the system training is done again
        print('SVM DEBUG')
        print(y_train.shape)
        svm_model_linear = SVC(kernel= kernel, C =C)
        svm_model_linear.fit(X_train, y_train)
        print('SVM DEBUG 2')
        svm_predictions = svm_model_linear.predict(X_test)

        return y_test, svm_predictions