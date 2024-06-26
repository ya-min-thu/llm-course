#%%
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nb import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train)
# %%
nb = NaiveBayes()
nb.fit(X_train, y_train)
pred = nb.predict(y_test)
# %%
accuracy(y_test, pred)

# %%
