import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target

Wiersze = X.shape[0]
Kolumny = X.shape[1]
Nazwy = len(np.unique(Y))

#minimalne wartosci kwiatow

Wartosc_Min = X.min(axis=0)
Wartosc_Max = X.max(axis=0)

#Normalizacja

Normalizacja_X = (X-Wartosc_Min)/(Wartosc_Max-Wartosc_Min)

#Ustawienia SOM, wartosci uczenia itd

np.random.seed(10)
liczba_neuronow = 3
ilosc_przejsc = 50
alpha_poczatkowe = 0.5

Waga = np.random.rand(liczba_neuronow,Kolumny)

#Algorytm SOM

for przejscie in range(ilosc_przejsc):
    alpha = alpha_poczatkowe / (1+przejscie)

    for kwiat in range(Normalizacja_X):

        odleglosci = np.zeros(ilosc_przejsc)

        for i in range(liczba_neuronow):
            roznica = kwiat - Waga[i]

            odleglosci[i] = np.sqrt(np.sum(roznica**2)) #euklides

    #wybranie neuronu z najmniejsza odlegloscia
    zwyciezca = np.argmin(odleglosci)

    #aktualizacja wag Neurony, ktory wygral
    Waga[zwyciezca] = Waga[zwyciezca] + alpha * (kwiat - Waga[zwyciezca])

#wyswietlenie wag po uczeniu
for i in range(ilosc_przejsc):
    print("Neuron",i+1,Waga[i])


print(Wiersze)
print(Kolumny)
print(Nazwy)
print(Wartosc_Min)
print(Wartosc_Max)
#print(X)
print(Normalizacja_X)