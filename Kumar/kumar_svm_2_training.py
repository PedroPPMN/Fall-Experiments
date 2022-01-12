import pickle
import pandas as pd 
import numpy as np 
import sklearn

from sklearn import svm
from sklearn.model_selection import train_test_split

tp = 0
fp = 0
fn = 0
tn = 0

filename = 'features.pickle'

def openfile(filename):
    with open(filename, 'rb') as infile:
        features_list = np.array(pickle.load(infile))
    return features_list

def standardization(X):
    X_means = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_means)/X_std
    return X

X = openfile(filename)[:, 0:3]
y = openfile(filename)[:,3]
X_train, X_test, y_train, y_test = train_test_split(standardization(X), y, test_size=0.3, random_state=43)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

for (y, p) in zip(y_test, pred):
    if y == 1 and p == 1:
        tp = tp + 1
    elif y == -1 and p == 1:
        fp = fp + 1
    elif y == 1 and p == -1:
        fn = fn + 1
    elif y == -1 and p == -1:
        tn = tn + 1
    else:
        raise Exception()
    
print()
print('Resultado Final:')
print(' -TPs = ' + str(tp))
print(' -FPs = ' + str(fp))
print(' -FNs = ' + str(fn))
print(' -TNs = ' + str(tn))
print(' -Sensitividade = ' + str(tp / (tp + fn + 1e-7)))
print(' -Especificidade = ' + str(tn / (tn + fp + 1e-7)))  