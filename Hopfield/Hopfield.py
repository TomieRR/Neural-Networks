import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

#seed
np.random.seed(1)

#Ustawienia
ilosc_neuronow = 64
max_ilosc_przejsc = 100
ilosc_obrazow = [1,3]
poziom_szumu = [0, 0.1, 0.2, 0.3]
metody = ["Hebb","Pseudoinwersja","Storkey"]

#=========================

cyfry = load_digits()
obrazy = cyfry.data
klasy = cyfry.target

obrazy_binarne = np.where(obrazy >= 8,1,-1)

for ile_obrazow in ilosc_obrazow:

    wzorce = []
    etykiety = []

    for cyfra in range(10):

        counter = 0

        for numer in range(len(klasy)):
            if klasy[numer] == ile_obrazow:
                wzorce.append(obrazy_binarne[numer])
                etykiety.append(cyfra)

                counter += 1

                if counter ==  ile_obrazow:
                    break

    wzorce = np.array(wzorce)
    etykiety = np.array(etykiety)

    print("Ilosc wzorców: ",len(wzorce))



