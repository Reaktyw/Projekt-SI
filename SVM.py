# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import glob


# # Lista plików CSV
# csv_files = glob.glob('fft_wyniki/fft_result_*.csv')

# # Załaduj dane z każdego pliku CSV
# data = []
# labels = []

# for i, file in enumerate(csv_files):
#     df = pd.read_csv(file)  # Wczytaj plik .csv
#     features = df.values.flatten()  # Zamień dane na jednowymiarową tablicę (features)
    
#     # Dodaj etykiety (np. 1 dla mężczyzny, 0 dla kobiety, zależnie od pliku)
#     # Tutaj musisz dostosować etykiety w zależności od tego, które pliki odpowiadają jakiej płci
#     label = 1 if i % 2 == 0 else 0  # Przykład, zmień to w zależności od danych
    
#     data.append(features)
#     labels.append(label)

# # Konwertuj listy na macierze numpy
# X = np.array(data)  # Features
# y = np.array(labels)  # Labels (płeć)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import glob
from sklearn.model_selection import cross_val_score


# Lista plików CSV
csv_files = glob.glob('fft_wyniki/fft_result_*.csv')

# Docelowa długość wektora
target_length = 5000

# Załaduj dane z każdego pliku CSV
data = []
labels = []
for i in range(50):
    labels.append(0)
for i in range(49):
    labels.append(1)

licznik = 0
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)  # Wczytaj plik .csv
    i = i - licznik

    
    try:
        df = df.applymap(lambda x: float(x) if isinstance(x, (int, float, str)) else np.nan)
    except ValueError:
        #print(f"Błąd konwersji w pliku: {file}")
        del labels[i]
        licznik = licznik + 1
        continue  # Pomijamy plik, jeśli zawiera błędne dane

    features = df.values.flatten()  # Zamień dane na jednowymiarową tablicę (features)







    # Jeśli długość cech jest mniejsza niż target_length, uzupełnij zerami
    if len(features) < target_length:
        features = np.pad(features, (0, target_length - len(features)), 'constant')
    # Jeśli długość cech jest większa niż target_length, przytnij do target_length
    elif len(features) > target_length:
        features = features[2:target_length]
    
    data.append(features)



# Konwertuj listy na macierze numpy
X = np.array(data)  # Features
y = np.array(labels)  # Labels (płeć)









# Sprawdzanie kształtu danych
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Tworzenie modelu SVM (klasyfikator)
model = SVC(kernel='linear', C=0.2)  # Używamy jądra liniowego (zwykle wystarczające w takich przypadkach)
#model = SVC(kernel='rbf', C=1.0, gamma='scale')
model = SVC(kernel='rbf', C=1.0, gamma=0.1)





# Walidacja krzyżowa (Cross-validation)
# Wykonaj 10-krotną walidację krzyżową
scores = cross_val_score(model, X, y, cv=5)  # cv=10 oznacza 10-krotną walidację krzyżową

print(scores)

# Średnia dokładność z walidacji krzyżowej
print(f"Średnia dokładność (10-krotna walidacja): {scores.mean() * 100:.2f}%")








# Podziel dane na dane treningowe i testowe (80% trening, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standaryzacja danych (przeskalowanie)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Użyj tylko na danych treningowych
X_test = scaler.transform(X_test)  # Użyj tych samych parametrów na danych testowych


# Trenowanie modelu
model.fit(X_train, y_train)

# Dokonaj predykcji na danych testowych
y_pred = model.predict(X_test)

# Oblicz dokładność modelu
accuracy = accuracy_score(y_test, y_pred)
print("y_pred na początku = ", y_pred)
print(f"Dokładność modelu na początku: {accuracy * 100:.2f}%")






train_accuracy = model.score(X_train, y_train)
print(f"Dokładność na zbiorze treningowym: {train_accuracy * 100:.2f}%")
