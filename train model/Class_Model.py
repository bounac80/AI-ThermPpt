#
#
# 
import joblib
import sklearn

from sklearn.base import BaseEstimator, RegressorMixin

class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)
