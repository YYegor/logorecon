# -*- coding: utf-8 -*-
# 12.01.2021, created by Egor Eremenko
import numpy as np
from pydub import AudioSegment
audio = AudioSegment.from_file('D:\\YandexDisk\\Logoped\\science\\logorecon\\voice\\voice_test.m4a')

print(np.array(audio.get_array_of_samples()))
