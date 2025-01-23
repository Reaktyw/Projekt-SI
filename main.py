import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

import os
import scipy
import csv
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def Fourier():
    # Pobieranie plików
    folder_path = 'kobiety'
    audio_extensions = ['.wav', '.mp3']
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]



    folder_path = 'mezczyzni'
    audio_extensions = ['.wav', '.mp3']
    files = files + [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]

    output_path = 'fft_wyniki'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rates = []
    aud_datas = []
    len_datas = []
    voices = []
    fouriers = []
    fouriers_to_plot = []
    w = []
    output_paths = []

    for i in range(len(files)):
        if i < 346:             # Wartość wpisana na sztywno, niestety nie mamy pomysłu jak to sensownie szybko zrobić
            folder_path = 'kobiety'
        else:
            folder_path = 'mezczyzni'

        # W zależności od rozszerzenia wykonuje się inny if
        file_path = os.path.join(folder_path, files[i])
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".wav":
            rate, aud_data = scipy.io.wavfile.read(file_path)
        elif ext == ".mp3":
            aud_data, rate = librosa.load(file_path, sr=None)  # sr=None oznacza brak resamplowania
            aud_data = (aud_data * 32767).astype(np.int16)  # Skaluje dane z zakresu -1..1 na 16-bitowe

        print('nr pliku = ', i,' ', aud_data.shape)

        # Jeżeli plik jest 2-kanałowy, wybieramy jeden kanał
        if len(aud_data.shape) != 1:
            aud_data = aud_data[:,0]

        rates.append(rate)
        aud_datas.append(aud_data)

        # Przygotowanie bufora o rozmiarze najbliższym potędze 2 - dla optymalizacji liczenia FFT
        len_datas.append(len(aud_data))
        voices.append(np.zeros(2**(int(np.ceil(np.log2(len_datas[i]))))))
        voices[i] = aud_datas[i]


        # Transformata fouriera
        fouriers.append(np.fft.fft(voices[i]))
        w.append(np.linspace(0, rates[i], len(fouriers[i])))
        fouriers_to_plot.append(fouriers[i][0:len(fouriers[i])//2]) # Odcinamy drugą połowę, bo jest symetryczna
        w[i] = w[i][0:len(fouriers[i])//2]


        # Normalizacja i zapis do plików do dalszej obróbki
        fourier_normalized = np.abs(fouriers_to_plot[i] / np.max(fouriers_to_plot[i]))
        
        output_path = f'fft_wyniki/fft_result_{i}.csv'
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for value in fourier_normalized:
                writer.writerow([value])  # Każda wartość w osobnym wierszu (jedna kolumna)







def SVM(X, y):
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score

    # Tworzenie modelu SVM (klasyfikator)
    model = SVC(kernel='linear', C=0.2)  # Używamy jądra liniowego

    # Walidacja krzyżowa (Cross-validation)
    scores = cross_val_score(model, X, y, cv=40)

    print(scores)

    # Średnia dokładność z walidacji krzyżowej
    print(f"Średnia dokładność (x-krotna walidacja): {scores.mean() * 100:.2f}%")


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
    print("y_pred = ", y_pred)
    print("y_test = ", y_test)
    print(f"Dokładność modelu na początku: {accuracy * 100:.2f}%")

    train_accuracy = model.score(X_train, y_train)
    print(f"Dokładność na zbiorze treningowym: {train_accuracy * 100:.2f}%")





def MLP(X,y):
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
    from tensorflow.keras.regularizers import l2

    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Konwersja etykiet do formatu one-hot
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model = Sequential([
        Dense(256, kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)), # l2 dodaje karę za zbyt duże wagi cech
        LeakyReLU(alpha=0.1), # Funkcja aktywacji, która dla wartości ujemnych nie ustawia wyjścia na 0, tylko 0.1
        Dropout(0.3), # Wyłącza losowo 30% wszystkich neuronów na każdy epoch
        Dense(128, kernel_regularizer=l2(0.01)),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(64, kernel_regularizer=l2(0.01)),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    # Kompilacja modelu
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Trenowanie modelu
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

    # Ocena modelu
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


    # Rysowanie wykresów wyników treningu
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Accuracy (train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation)')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Loss (train)')
    plt.plot(history.history['val_loss'], label='Loss (validation)')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()








def Autogluon(X,y):
    from tensorflow.keras.utils import to_categorical
    from autogluon.tabular import TabularPredictor

    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Przygotowanie danych do TabularPredictor (pandas DataFrame)
    train_data = pd.DataFrame(X)
    train_data['target'] = y

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Przygotowanie etykiet w formacie one-hot [[1,0],[0,1],itd.]
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Przygotowanie danych testowych do DataFrame (w celu kompatybilności z AutoGluon)
    test_data = pd.DataFrame(X_test)
    test_data['target'] = y_test.argmax(axis=1)  # Zamieniamy one-hot na wartości klasy

    predictor = TabularPredictor(label='target', problem_type='binary', verbosity=2)

    predictor.fit(train_data)

    # Wyświetlenie wyników
    leaderboard = predictor.leaderboard()
    print(leaderboard)


    y_pred = predictor.predict(test_data)

    comparison = pd.DataFrame({ # Tworzy tabelę z predykcjami i rzeczywistymi wartościami
        'y_pred': y_pred,
        'y_test': test_data['target']
    })

    print(comparison)

    accuracy = predictor.evaluate(test_data)
    print(f'Finalna dokładność modelu AutoGluon: {accuracy["accuracy"]:.4f}')






if __name__ == "__main__":
    odpowiedz = input("Czy wykonać transformatę Fouriera? (tak/nie) ").strip().lower()

    if odpowiedz == "tak":
        Fourier()


    csv_files = glob.glob('fft_wyniki/fft_result_*.csv')

    X = []
    y = []

    licznik_bledow = 0
    target_length = 10000

    # Przygotowanie danych
    for i, file in enumerate(csv_files):
        fft_data = pd.read_csv(file)

        # Jeżeli plik ma błędne dane -> usuwamy go i cofamy "i" o jeden
        i = i - licznik_bledow
        if i%100 == 0:
            print(f"{i} plików jest przygotowanych.")
        try:
            treshold = fft_data.max().max()/2
            fft_data = fft_data.map(lambda x: float(x) if isinstance(x, (int, float, str)) else np.nan)
        except ValueError:
            del y[i]
            licznik_bledow += 1
            continue  # Pomijamy plik, jeśli zawiera błędne dane

        features = fft_data.values.flatten()  # Zamień dane na jednowymiarową tablicę (features)

        # Uzupełnienie lub przycięcie danych do target_length
        if len(features) < target_length:
            features = np.pad(features, (0, target_length - len(features)), 'constant')
        elif len(features) > target_length:
            features = features[0:target_length]

        X.append(features)

    # Przygotowanie labeli odpowiednio 0-kobieta, 1-mężczyzna
    for i in range(346):
        y.append(0)
    for i in range(346):
        y.append(1)


    # Przekształcenie do numpy array
    X = np.array(X)
    y = np.array(y)

    odpowiedz = input("Który model wybierasz? (Autogluon, MLP, SVM) ").strip().lower()

    match odpowiedz:
        case "autogluon":
            Autogluon(X,y)
        case "svm":
            SVM(X,y)
        case "mlp":
            MLP(X,y)
        case _:
            print("Nieznana komenda.")