# IEMOCAP Emotion Recognition

# About IEMOCAP
The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is an acted, multimodal and multispeaker database, recently collected at SAIL lab at USC. It contains approximately 12 hours of audiovisual data, including video, speech, motion capture of face, text transcriptions. It consists of dyadic sessions where actors perform improvisations or scripted scenarios, specifically selected to elicit emotional expressions. IEMOCAP database is annotated by multiple annotators into categorical labels, such as anger, happiness, sadness, neutrality, as well as dimensional labels such as valence, activation and dominance.
You can get the dataset from here. (By registering on [this page](https://sail.usc.edu/iemocap/release_form.php) 
https://sail.usc.edu/iemocap/ 

Now In the dataset received, the wav files ane in 2 formats.
1. In dialogue format
2. In sentence format 

In using the dialogue format the wav file generate a large number of audio vectors and it is not fesiable to use it because of memory constraints. So that, in simple way, I have copied all the sentence wav file in a single folder which makes code easier and make wav files easier to access.

I have take 4 emotions namely Anger, Sadness, Happiness and Sadness and tried to separate the python files or jupyter notebook for small tasks like, extracting the audio feature vectors, generating the spectrograms, and finally use 1D CNN to recognize the emotion. 
