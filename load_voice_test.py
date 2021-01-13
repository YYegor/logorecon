# -*- coding: utf-8 -*-
# 12.01.2021, created by Egor Eremenko
import numpy as np
#from pydub import AudioSegment
import matplotlib.pyplot as plt

#audio = AudioSegment.from_file('D:\\YandexDisk\\Logoped\\science\\logorecon\\voice\\voice_test.m4a')
#adata = np.array(audio.get_array_of_samples())

import IPython.display as ipd
import librosa

import librosa.display
filesample = 'D:\\YandexDisk\\Logoped\\science\\logorecon\\voice\\voice_test.m4a'
#ipd.Audio(filesample)
plt.figure(figsize=(15,4))
data, sample_rate1 = librosa.load(filesample, sr=22050, mono=True, offset=0.0, duration=50, res_type='kaiser_best')
librosa.display.waveplot(data,sr=sample_rate1, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000)
plt.show()
print(sample_rate1)

from scipy import signal
from scipy.fft import fftshift
f, t, Sxx = signal.spectrogram(data, sample_rate1)

plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()