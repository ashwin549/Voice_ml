import pickle
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
import math
import os
from os import walk

#-------------------------------Splitting audio file into 3s bits-------------------------------------#
class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splited successfully')

#----------------------------------Extracting features from audio------------------------------------------#
def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    return result


#----------------------------------IMPORT AUDIO FILES-----------------------------------------#
folder = "C:\\Users\\Ashwin\\Desktop\\code\\splittingaudio_test"                   #folder where the splitted audio will be there
file = "stone-construction-work-60972.wav"                                         #audio file. Must be in same folder as above
split_wav = SplitWavAudioMubin(folder, file)
split_wav.multiple_split(sec_per_split=3)

paths = []
for (dirpath, dirnames, filenames) in walk(folder):
    paths.extend(filenames)
    break
x= []
count=0
for path in paths:
    if path!=file:
        filedur=SplitWavAudioMubin(folder, path)
        duration=filedur.get_duration()
        if(duration>1):
            x.append([])
            feature = get_features(os.path.join(folder, path))
            for ele in feature:
                x[count].append([ele])
            count+=1

x=np.array(x)
length=len(x)

#--------------------------------------------For predicting from single 3s clip----------------------------------------#
#input=r"C:\Users\Ashwin\Desktop\code\Voice_ML\SAVEE\ALL\DC_h02.wav"     #Enter path of audio file here
#x=[]
#feature=get_features(input)
#for ele in feature:
#    x.append([ele])
#x=[x]
#x=np.array(x)
#----------------------------------------------------------------------------------------------------------------------#

#Unloading the CNN model and rebuilding it
model_pkl_file = "C:/Users/Ashwin/Desktop/code/Voice_ML/Voice_emotion_detection.pkl" 
with open(model_pkl_file, 'rb') as file:  
    model2 = pickle.load(file)

#predicting...
y_predict = model2.predict(x)

#Getting proper output from y_predict    
encoded_list=['angry','calm','disgust','fear','happy','neutral','sad','surprise']
for i in range (0,length,1):
    index = np.argmin(np.abs(y_predict-1.0000000))
    emotion=encoded_list[index]
    print(emotion)