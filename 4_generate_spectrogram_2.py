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

print(labels_df.emotion.value_counts())
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

labels_df_subset = labels_df[labels_df["emotion"].isin(["neu", 'ang', 'hap', 'sad'])]

labels_df_subset = labels_df_subset[["wav_file", "emotion"]]

labels_df_subset = labels_df_subset.rename(columns={"wav_file": "filename"})

pickle_to_df = pd.read_csv('/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/pickle_to_audiodf/pickle_to_audiodf_5.csv')

del pickle_to_df['Unnamed: 0']

pickle_to_subset = labels_df_subset.merge(pickle_to_df, on="filename",right_index=True)

print(pickle_to_subset.shape)
print(pickle_to_df.shape)
print(labels_df_subset.shape)

pickle_to_subset.to_csv('/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/Preprocessing/pickle_to_subset/pickle_to_subset_5.csv',index = False)