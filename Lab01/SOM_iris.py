from enum import unique

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target
W = np.random.rand(3,4)

Wiersze = X.shape[0]
Kolumny = X.shape[1]
Nazwy = len(np.unique(Y))

#minimalne wartosci kwiatow
Wartosc_Min = X.min(axis=0)
Wartosc_Max = X.max(axis=0)

#standaryzacja

Standaryzacja = (X-Wartosc_Min)/(Wartosc_Max-Wartosc_Min)

print(Wiersze)
print(Kolumny)
print(Nazwy)
print(Wartosc_Min)
print(Wartosc_Max)
#print(X)
print(Standaryzacja)