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

labels_df = pd.read_csv('/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/df_iemocap/df_iemocap_5.csv')
iemocap_dir = '/home/mds-student/Documents/aDITYA/spec_augment-master/IEMOCAP_full_release_withoutVideos/Iemocap_sentences/'
save_dir = '/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/AudioVectors/'
audio_vectors_path= save_dir + 'audio_vectors_'

pickle_to_df = pd.DataFrame(columns=["filename", "audio_vector"])

for sess in [5]:
    audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
    for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
        wav_file_name = (row['wav_file'])
        y = audio_vectors[wav_file_name]
        list_y = list(y)
        pickle_to_df = pickle_to_df.append({'filename': wav_file_name, 'audio_vector': list_y}, ignore_index=True)

pickle_to_df.to_csv('/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/pickle_to_audiodf/pickle_to_audiodf_5.csv')