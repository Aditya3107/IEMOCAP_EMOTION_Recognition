import numpy as np
import pandas as pd
import os.path
import librosa
import IPython.display as ipd
from shutil import copyfile
import glob
import matplotlib.style as ms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random
import time

import IPython.display
import librosa.display
import joblib
from joblib import Parallel, delayed
from PIL import Image
import multiprocessing as mp

import pandas as pd
import math

labels_df = pd.read_csv('/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/df_iemocap/df_iemocap_1.csv')
iemocap_dir = '/home/mds-student/Documents/aDITYA/spec_augment-master/IEMOCAP_full_release_withoutVideos/Iemocap_sentences/'
save_dir = '/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/AudioVectors/'
audio_vectors_path= save_dir + 'audio_vectors_'

emotion_dict = {'ang': 0,
                'hap': 1,
                'exc': 2,
                'sad': 3,
                'fru': 4,
                'fea': 5,
                'sur': 6,
                'neu': 7,
                'xxx': 8,
                'oth': 8,
                'dis': 8}

emotion_full_dict = {'neu': 'neutral', 'ang': 'anger', 'hap': 'happiness', 'exc': 'excited', 'sad': 'sadness',
                     'fru':'frustrated', 'fea': 'fear', 'sur': 'surprised', 'xxx': 'others', 'oth': 'others', 'dis': 'others'}

pickle_to_subset = pd.read_csv('/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/pickle_to_subset/pickle_to_subset_1.csv')

pathAudio = iemocap_dir
pathImage = '/home/mds-student/Documents/aDITYA/spec_augment-master/IEMOCAP_full_release_withoutVideos/Iemocap_sentences/SpecAugment_spectrograms/'

#pickle_to_subset = pickle_to_subset[]

sample_rate = 44100
import time
import multiprocessing as mp
import ast
for row in tqdm(pickle_to_subset.itertuples(index=True, name='Pandas')):
    a = getattr(row, "audio_vector")
    image_fname = getattr(row, "filename")
    label = getattr(row, "emotion")
    res = ast.literal_eval(a)
    res = np.asarray(res)
    res = res.astype('float32')
    S = librosa.feature.melspectrogram(y = res, sr = sample_rate, n_mels=256,hop_length=128,fmax=8000)
    S_2 = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12,4))
    librosa.display.specshow(S_2, sr=sample_rate, x_axis='time', y_axis='mel')
    #plt.title('mel power spectrogram')
    #plt.colorbar(format='%+02.0f dB')
    #plt.tight_layout()
    fig1 = plt.gcf()
    plt.axis('off')
    save_path = pathImage + emotion_full_dict[label] + '/' + image_fname + '.jpg'
    fig1.savefig(save_path, dpi=50)
    plt.close(fig1)