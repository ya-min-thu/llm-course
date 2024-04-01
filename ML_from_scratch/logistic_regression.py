import numpy as np
from linear_regresssion import BaseRegression

class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)

    def predict(self, X, w, b):
        linear_model = np.dot(X, w)  + b
        y_predicted = self._sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))