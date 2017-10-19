
**Kacper Radzikowski**  
**Michał Przysucha**

###Współczene techniki heurystyczne (WMH) – projekt
###PB2. Zastosowanie sieci neuronowej w zadaniu aproksymacyjnym

Należy zaprojektować sztuczną sieć neuronową do aproksymacji ciągłej funkcji dwuwymiarowej. Sieć powinna nauczyć się określonej liczby punktów, a następnie prawidłowo
(z akceptowalnym błędem) aproksymować punkty z drugiego zbioru (testowego). Projekt powinine obejmować wypróbowanie różnych struktur sieci (liczba warstw ukrytych
oraz liczba neuronów w warstwie) oraz określenie struktury optymalnej dla problemu. Należy też zbadać proces uczenia się sieci.

Implementacja - podział prac:  
1. Wygenerowanie zbioru danych - Kacper  
2. Inicjalizacja parametrów - Kacper  
3. Forward propagation - Kacper  
4. Policzenie aktualnego kosztu - Kacper  
5. Backprop - Michał  
6. Update parametrów - Michał  
7. Fukncja główna (spinająca 5 powyższych) - Michał  
8. Wizualizacja kosztu jako funkcji od numeru iteracji - Michał  

Dodatkowo warto policzyć różne metryki:  
- błąd na zbiorze trenującym (czy występuje bias - underfitting)  
- błąd na zbiorze developerskim (czy wystepuje variance - overfitting)  
- ewaluacja na zbiorze testowym  
 
