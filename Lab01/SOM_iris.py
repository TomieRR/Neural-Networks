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

# Ustawienia SOM

liczba_neuronow = 3

# Testy

Epoki_test = [50, 10, 100, 50, 50, 50, 50]
Alpha_test = [0.5, 0.5, 0.5, 0.1, 0.9, 0.5, 0.5]
Funkcje_test = [
    "odwrotna", "odwrotna", "odwrotna",
    "odwrotna", "odwrotna", "wykladnicza", "liniowa"
]

for test in range(len(Epoki_test)):

    ilosc_przejsc = Epoki_test[test]
    alpha_poczatkowe = Alpha_test[test]
    funkcja = Funkcje_test[test]

    # Kazdy test zaczyna od tych samych wag

    np.random.seed(10)
    Waga = np.random.rand(liczba_neuronow, Kolumny)

    print("\nTest:", test + 1)
    print("Epoki:", ilosc_przejsc)
    print("Alpha poczatkowe:", alpha_poczatkowe)
    print("Funkcja:", funkcja)

    # Algorytm SOM

    for przejscie in range(ilosc_przejsc):

        if funkcja == "odwrotna":
            alpha = alpha_poczatkowe / (1 + przejscie)

        elif funkcja == "wykladnicza":
            alpha = alpha_poczatkowe * np.exp(-0.1 * przejscie)

        elif funkcja == "liniowa":
            alpha = alpha_poczatkowe * (1 - przejscie / ilosc_przejsc)

        if (przejscie + 1) % 10 == 0:
            print("Epoka:", przejscie + 1, "Alpha:", alpha)

        for kwiat in Normalizacja_X:
            odleglosci = np.zeros(liczba_neuronow)

            for i in range(liczba_neuronow):
                roznica = kwiat - Waga[i]
                odleglosci[i] = np.sqrt(np.sum(roznica**2))

            # Wybor zwyciezcy
            zwyciezca = np.argmin(odleglosci)

            # Aktualizacja wag
            Waga[zwyciezca] += alpha * (kwiat - Waga[zwyciezca])

    # Wagi po zakonczeniu uczenia

    for i in range(liczba_neuronow):
        print("Neuron", i + 1, Waga[i])

    # Przypisywanie kazdego kwiatu do neuronu

    Przypisania = np.zeros(Wiersze, dtype=int)

    for numer, kwiat in enumerate(Normalizacja_X):
        odleglosci = np.zeros(liczba_neuronow)

        for i in range(liczba_neuronow):
            roznica = kwiat - Waga[i]
            odleglosci[i] = np.sqrt(np.sum(roznica**2))

        zwyciezca = np.argmin(odleglosci)
        Przypisania[numer] = zwyciezca

        print("Kwiat", numer + 1, "-> Neuron", zwyciezca + 1)

    # Macierz

    Macierz = np.zeros((liczba_neuronow, Nazwy), dtype=int)

    for numer in range(Wiersze):
        neuron = Przypisania[numer]
        klasa = Y[numer]

        Macierz[neuron, klasa] += 1

    Tabela = pd.DataFrame(
        Macierz,
        index=["N1", "N2", "N3"],
        columns=iris.target_names)

    print("\nMacierz neurony i klasy:")
    print(Tabela)

#print(Wiersze)
#print(Kolumny)
#print(Nazwy)
#print(Wartosc_Min)
#print(Wartosc_Max)
#print(X)
#print(Normalizacja_X)