import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import glob

csv_files = glob.glob('fft_wyniki/fft_result_*.csv')

X = []
y = []
for i in range(50):
    y.append(0)
for i in range(49):
    y.append(1)

licznik_bledow = 0
target_length = 20000

for i, file in enumerate(csv_files):
    fft_data = pd.read_csv(file)
    i = i - licznik_bledow

    try:
        fft_data = fft_data.applymap(lambda x: float(x) if isinstance(x, (int, float, str)) else np.nan)
    except ValueError:
        del y[i]
        licznik_bledow = licznik_bledow + 1
        continue  # Pomijamy plik, jeśli zawiera błędne dane

    features = fft_data.values.flatten()  # Zamień dane na jednowymiarową tablicę (features)

    # Jeśli długość cech jest mniejsza niż target_length, uzupełnij zerami
    if len(features) < target_length:
        features = np.pad(features, (0, target_length - len(features)), 'constant')
    # Jeśli długość cech jest większa niż target_length, przytnij do target_length
    elif len(features) > target_length:
        features = features[:target_length]

    X.append(features)


X = np.array(X)
y = np.array(y)


# Normalizacja danych
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konwersja etykiet do formatu one-hot
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Budowa modelu
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Ocena modelu
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')