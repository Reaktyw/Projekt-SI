#--------------JAK SIĘ ZACZĘŁO---------------


# import numpy as np
# import matplotlib.pyplot as plt
# import scipy

# file = 'C:/Wszystko nowe/Visual Studio Code/VisualStudioCode/Projekty/moj_stuff/dzwieki/the night sky.wav'
# rate, aud_data = scipy.io.wavfile.read(file)
# # From here down, everything else can be the same
# len_data = len(aud_data)

# channel_1 = np.zeros(2**(int(np.ceil(np.log2(len_data)))))
# channel_2 = np.zeros(2**(int(np.ceil(np.log2(len_data)))))
# channel_1[0:len_data] = aud_data[:,0]                       # Pierwszy kanał
# channel_2[0:len_data] = aud_data[:,1]                       # Drugi kanał

# #PIERWSZY KANAŁ
# fourier1 = np.fft.fft(channel_1)
# w1 = np.linspace(0, rate, len(fourier1))
# # Odcinamy drugą połowę, bo jest symetryczna
# fourier_to_plot1 = fourier1[0:len(fourier1)//2]
# w1 = w1[0:len(fourier1)//2]
# plt.figure(1)
# plt.plot(w1, fourier_to_plot1)

# #DRUGI KANAŁ
# fourier2 = np.fft.fft(channel_2)
# w2 = np.linspace(0, rate, len(fourier2))

# fourier_to_plot2 = fourier2[0:len(fourier2)//2]
# w2 = w2[0:len(fourier2)//2]


# plt.plot(w2, fourier_to_plot2)
# plt.xlabel('frequency')
# plt.ylabel('amplitude')
# plt.show()






#--------------PIERWSZE GŁOSY---------------

# import numpy as np
# import matplotlib.pyplot as plt
# import scipy

# file1 = 'dzwieki/glos_meski_1.wav'
# file2 = 'dzwieki/glos_kobiecy_1.wav'
# rate1, aud_data1 = scipy.io.wavfile.read(file1)
# rate2, aud_data2 = scipy.io.wavfile.read(file2)

# len_data1 = len(aud_data1)
# len_data2 = len(aud_data2)

# channel_1 = np.zeros(2**(int(np.ceil(np.log2(len_data1)))))
# channel_2 = np.zeros(2**(int(np.ceil(np.log2(len_data2)))))
# channel_1[0:len_data1] = aud_data1                       # Pierwszy głos
# channel_2[0:len_data2] = aud_data2                       # Drugi głos

# #PIERWSZY GŁOS
# fourier1 = np.fft.fft(channel_1)
# w1 = np.linspace(0, rate1, len(fourier1))
# # Odcinamy drugą połowę, bo jest symetryczna
# fourier_to_plot1 = fourier1[0:len(fourier1)//2]
# w1 = w1[0:len(fourier1)//2]
# plt.figure(1)
# plt.plot(w1, fourier_to_plot1)
# plt.xlabel('frequency')
# plt.ylabel('amplitude')

# # DRUGI GŁOS
# fourier2 = np.fft.fft(channel_2)
# w2 = np.linspace(0, rate2, len(fourier2))
# fourier_to_plot2 = fourier2[0:len(fourier2)//2]
# w2 = w2[0:len(fourier2)//2]
# plt.figure(2)
# plt.plot(w2, fourier_to_plot2)
# plt.xlabel('frequency')
# plt.ylabel('amplitude')

# plt.show()







#--------------ZROBIENIE Z GŁOSÓW LISTY---------------

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv

# Ścieżka do folderu, w którym są pliki
folder_path = 'dzwieki'
# Pobranie wszystkich plików o rozszerzeniach .wav
audio_extensions = ['.wav']
files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]


rates = []
aud_datas = []
len_datas = []
voices = []
fouriers = []
fouriers_to_plot = []
w = []
output_paths = []

for i in range(len(files)):
    #Ładowanie plików i robienie rzeczy potrzebnych do fft
    file_path = os.path.join(folder_path, files[i])
    rate, aud_data = scipy.io.wavfile.read(file_path)

    rates.append(rate)
    aud_datas.append(aud_data)

    len_datas.append(len(aud_data))
    voices.append(np.zeros(2**(int(np.ceil(np.log2(len_datas[i]))))))
    voices[i] = aud_datas[i]


    # Transformata fouriera
    fouriers.append(np.fft.fft(voices[i]))
    w.append(np.linspace(0, rates[i], len(fouriers[i])))
    fouriers_to_plot.append(fouriers[i][0:len(fouriers[i])//2]) # Odcinamy drugą połowę, bo jest symetryczna
    w[i] = w[i][0:len(fouriers[i])//2]

    # Wyświetlanie transformat każdego głosu na osobnym plocie
    plt.figure(files[i])
    plt.plot(w[i], fouriers_to_plot[i])
    plt.xlabel('frequency')
    plt.ylabel('amplitude')


    # Normalizacja i zapis do plików do dalszej obróbki
    fourier_normalized = np.abs(fouriers_to_plot[i] / np.max(fouriers_to_plot[i]))
    
    output_path = f'fft_wyniki/fft_result_{i}.csv'
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for value in fourier_normalized:
            writer.writerow([value])  # Każda wartość w osobnym wierszu (jedna kolumna)

plt.show()