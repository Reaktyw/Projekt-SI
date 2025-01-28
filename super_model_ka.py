import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import scipy
import csv
import librosa
import numpy as np
import glob
import re
from sklearn.preprocessing import StandardScaler


def Labels():
    folder_path = 'nowe_dzwieki'
    audio_extensions = ['.wav', '.mp3']
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]
    labels = []
    for name in files:
        if "kobiecy" in name:
            labels.append("Kobieta")
        elif "meski" in name:
            labels.append("Mężczyzna")
    return labels


def Fourier():
    folder_path = 'nowe_dzwieki'
    audio_extensions = ['.wav', '.mp3']
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]

    output_path = 'fft_wyniki_nowe'
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
        file_path = os.path.join(folder_path, files[i])
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".wav":
            rate, aud_data = scipy.io.wavfile.read(file_path)
        elif ext == ".mp3":
            aud_data, rate = librosa.load(file_path, sr=None)
            aud_data = (aud_data * 32767).astype(np.int16)

        print('Nr pliku = ', i,' ', aud_data.shape)

        if len(aud_data.shape) != 1:
            aud_data = aud_data[:,0]

        rates.append(rate)
        aud_datas.append(aud_data)

        len_datas.append(len(aud_data))
        voices.append(np.zeros(2**(int(np.ceil(np.log2(len_datas[i]))))))
        voices[i] = aud_datas[i]

        fouriers.append(np.fft.fft(voices[i]))
        w.append(np.linspace(0, rates[i], len(fouriers[i])))
        fouriers_to_plot.append(fouriers[i][0:len(fouriers[i])//2])
        w[i] = w[i][0:len(fouriers[i])//2]

        fourier_normalized = np.abs(fouriers_to_plot[i] / np.max(fouriers_to_plot[i]))
        
        output_path = f'fft_wyniki_nowe/fft_result_{i}.csv'
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for value in fourier_normalized:
                writer.writerow([value])


if __name__ == "__main__":
    odpowiedz = input("Czy wykonać transformatę Fouriera? (tak/nie) ").strip().lower()

    if odpowiedz == "tak":
        Fourier()
    
    # Ścieżka do modelu
    model_path = r"AutogluonModels/ag-20250124_014210"
    #predictor = TabularPredictor.load(model_path)
    predictor = TabularPredictor.load(model_path, require_py_version_match=False)

    # Folder z plikami do predykcji
    csv_files = glob.glob('fft_wyniki_nowe/fft_result_*.csv')
    csv_files = sorted(csv_files, key=lambda x: int(re.search(r'fft_result_(\d+)', x).group(1)))

    target_length = 10000
    X = []
    print("Za moment pojawi się wynik :)")
    # Przetwarzanie każdego pliku
    for i, file in enumerate(csv_files):
        fft_data = pd.read_csv(file)
        try:
            fft_data = fft_data.map(lambda x: float(x) if isinstance(x, (int, float, str)) else np.nan)
        except ValueError:
            continue

        features = fft_data.values.flatten()  # Zamień dane na jednowymiarową tablicę (features)

        # Uzupełnienie lub przycięcie danych do target_length
        if len(features) < target_length:
            features = np.pad(features, (0, target_length - len(features)), 'constant')
        elif len(features) > target_length:
            features = features[0:target_length]

        X.append(features)

    X = np.array(X)

    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Przygotowanie danych do TabularPredictor (pandas DataFrame)
    test_data = pd.DataFrame(X)
    
    predictions = predictor.predict(test_data)

    labels = Labels()

    print("Plik                               Predykcja Rzeczywistość")
    for i, predicted_class in enumerate(predictions):
        if predicted_class == 0:
            print(f"{csv_files[i]}   Kobieta   {labels[i]}")
        elif predicted_class == 1:
            print(f"{csv_files[i]}   Mężczyzna  {labels[i]}")



    # Oblicz dokładność
    true_classes = [0 if label == "Kobieta" else 1 for label in labels]
    correct_predictions = np.sum(predictions == true_classes)  # Liczba poprawnych predykcji
    total_predictions = len(predictions)  # Całkowita liczba przykładów
    accuracy = correct_predictions / total_predictions  # Dokładność jako ułamek

    print(f"\nDokładność modelu: {accuracy * 100:.2f}%")
