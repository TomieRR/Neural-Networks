import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

#seed
np.random.seed(1)

#Ustawienia
ilosc_neuronow = 64
ilosc_przejsc = 100
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
            if klasy[numer] == cyfra:
                wzorce.append(obrazy_binarne[numer])
                etykiety.append(cyfra)

                counter += 1

                if counter ==  ile_obrazow:
                    break

    wzorce = np.array(wzorce)
    etykiety = np.array(etykiety)

    print("Ilosc wzorców: ",len(wzorce))

#=========
    zbiory = []

    for szum in poziom_szumu:

        zaszumione = wzorce.copy()

        ile_pikseli = int(ilosc_neuronow * szum)

        for numer in range(len(wzorce)):

            indeksy = np.random.choice(ilosc_neuronow, ile_pikseli, replace=False)

            for i in indeksy:
                zaszumione[numer, i] = zaszumione[numer,i] * (-1)

        zbiory.append(zaszumione)

    #porownanie

    for metoda in metody:

        waga = np.zeros((ilosc_neuronow, ilosc_neuronow))

# Hebb

        if metoda == "Hebb":

            for wzorzec in wzorce:

                for i in range(ilosc_neuronow):
                    for j in range(ilosc_neuronow):

                        if i != j:
                            waga[i, j] += wzorzec[i] * wzorzec[j]

            waga = waga / ilosc_neuronow

#Pseudoinwersja

        elif metoda == "Pseudoinwersja":

            pseudoodwrotna = np.linalg.pinv(wzorce)
            waga = pseudoodwrotna @ wzorce

#Storkey

        elif metoda == "Storkey":

            for wzorzec in wzorce:

                stara_waga = waga.copy()

                for i in range(ilosc_neuronow):
                    for j in range(i+1, ilosc_neuronow):

                        pole_i = 0
                        pole_j = 0


                        for bez in range(ilosc_neuronow):

                            if bez != i and bez != j:
                                pole_i += stara_waga[i, bez] * wzorzec[bez]
                                pole_j += stara_waga[j, bez] * wzorzec[bez]

                        zamiana = (
                            wzorzec[i] * wzorzec[j]
                            - wzorzec[i] * pole_j
                            - pole_i * wzorzec[j]
                        ) / ilosc_neuronow

                        waga[i,j] = stara_waga[i,j] + zamiana
                        waga[j,i] = waga[i,j]

        np.fill_diagonal(waga, 0)
        print("\nMetoda:", metoda)
        print("Symetria:", np.allclose(waga, waga.T))
        print("Zera na przekatnej:", np.all(np.diag(waga) == 0))

# sprawdzenie kazdego poziomu szumu

        for test in range(len(poziom_szumu)):

            zaszumione = zbiory[test]
            poprawne = 0

            for numer in range(len(wzorce)):

                stan = zaszumione[numer].copy()

#Jeden obraz

                for przejscie in range(ilosc_przejsc):

                    poprzedni = stan.copy()

                    for i in range(ilosc_neuronow):

                        pole = 0

                        for j in range(ilosc_neuronow):
                            pole += waga[i, j] * stan[j]

                        if pole > 0:
                            stan[i] = 1
                        elif pole < 0:
                            stan[i] = -1


                    if np.array_equal(stan, poprzedni):
                        break


                if np.array_equal(stan, wzorce[numer]):
                    poprawne += 1



#wizualizacja, porownanie

                if numer == 0:
                    plt.figure(figsize=(9, 3))

                    plt.subplot(1, 3, 1)
                    plt.imshow(wzorce[numer].reshape(8, 8), cmap="gray")
                    plt.title("Oryginal")

                    plt.subplot(1, 3, 2)
                    plt.imshow(zaszumione[numer].reshape(8, 8), cmap="gray")
                    plt.title("Szum")

                    plt.subplot(1, 3, 3)
                    plt.imshow(stan.reshape(8, 8), cmap="gray")
                    plt.title("Wynik")

                    plt.suptitle(
                        f"{metoda}, szum: {poziom_szumu[test] * 100}%"
                    )
                    plt.tight_layout()
                    plt.show()

                print("Szum:", poziom_szumu[test] * 100, "%")
                print("Odtworzone obrazy:", poprawne, "/", len(wzorce))