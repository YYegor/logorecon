# -*- coding: utf-8 -*-
# 12.01.2021, created by Egor Eremenko
import numpy as np
#from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
import librosa.display




#audio = AudioSegment.from_file('D:\\YandexDisk\\Logoped\\science\\logorecon\\voice\\voice_test.m4a')
#adata = np.array(audio.get_array_of_samples())


filesample_defect = 'voice\\трескучие январские морозы - дефект.m4a'
filesample_norm = 'voice\\трескучие январские морозы - норма.m4a'

#ipd.Audio(filesample)
plt.figure(figsize=(15,4))
signal, sample_rate1 = librosa.load(filesample_norm, sr=11025, mono=True, offset=0.0, duration=50, res_type='kaiser_best')
signal_d, sample_rate1_d = librosa.load(filesample_defect, sr=22050, mono=True, offset=0.0, duration=50, res_type='kaiser_best')

librosa.display.waveplot(signal, sr=sample_rate1, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000)
librosa.display.waveplot(signal_d, sr=sample_rate1, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000)

#plt.show()
print(sample_rate1)

#mfcc part
mfcc = librosa.feature.mfcc(signal, n_mfcc=13, sr=sample_rate1)

print(mfcc)

cr = mfcc[1:350]
cr2 = mfcc[351:751]



print(mfcc.size)
print(mfcc.ctypes)
print(cr)

print(cr.size)

print(cr.ctypes)
mfcc_d = librosa.feature.mfcc(signal_d, n_mfcc=13, sr=sample_rate1_d)

print(mfcc_d.size)

plt.figure(figsize=(25, 10))
librosa.display.specshow(mfcc, x_axis="time", sr=sample_rate1)
plt.colorbar(format="%+2f")
plt.show()

plt.figure(figsize=(25, 10))
librosa.display.specshow(cr, x_axis="time", sr=sample_rate1)
plt.colorbar(format="%+2f")
plt.show()




plt.figure(figsize=(25, 10))
librosa.display.specshow(mfcc_d, x_axis="time", sr=sample_rate1_d)
plt.colorbar(format="%+2f")
plt.show()

exit()


f, t, Sxx = signal.spectrogram(signal, sample_rate1)

plt.pcolormesh(t, f, Sxx, shading='gouraud')

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.figure(figsize=(15,4))
signal, sample_rate1 = librosa.load(filesample_norm, sr=44100, mono=True, offset=0.0, duration=50, res_type='kaiser_best')
librosa.display.waveplot(signal, sr=sample_rate1, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000)
plt.title = filesample_norm
plt.show()
print(sample_rate1)





f, t, Sxx = signal.spectrogram(signal, sample_rate1)

plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()