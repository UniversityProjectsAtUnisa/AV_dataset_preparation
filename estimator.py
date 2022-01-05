import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC


class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, low_black, high_black, *args, **kwargs):
        self.model = SVC(*args, **kwargs)
        self.low_black = low_black
        self.high_black = high_black
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_ = multiple_preprocessing(X, self.low_black, self.high_black)
        self.y_ = y

        self.model.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = multiple_preprocessing(X, self.low_black, self.high_black)

        y = self.model.predict(X)
        return y
