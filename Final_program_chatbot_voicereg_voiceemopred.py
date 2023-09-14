import speech_recognition as sr
import pyttsx3
import pickle
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
import torch.nn as nn
import nltk
nltk.download('punkt')
from os import walk
import os, shutil
import threading
import random
import json

filename=r"C:\Users\Ashwin\Desktop\code\Voice_ML\intents.json"
with open(filename, 'r') as f:
    intents = json.load(f)
model_pkl_file = "C:/Users/Ashwin/Desktop/code/Voice_ML/Voice_emotion_detection.pkl" 
with open(model_pkl_file, 'rb') as file:  
    model2 = pickle.load(file)
emotion_list=[]

#-----------------------------------Emptying the contents of the folder-------------------------------#
def empty():
    folder_to_be_emptied = "C:\\Users\\Ashwin\\Desktop\\code\\Voice_ML\\voice"
    for filename in os.listdir(folder_to_be_emptied):
        file_path = os.path.join(folder_to_be_emptied, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#-------------------------------------Voice recognition part--------------------------------------------#
def SpeakText(command):
        
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
    
def voice_recog():
    # Initialize the recognizer
    r = sr.Recognizer()
    # Function to convert text to
    # speech
    a=1

    while a==1:   
        # Exception handling to handle
        # exceptions at the runtime
        try:
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)
                #print("Speak now:") 
                #listens for the user's input
                audio2 = r.listen(source2)
                with open('C:/Users/Ashwin/Desktop/code/Voice_ML/voice/speech.wav','wb') as f:
                    f.write(audio2.get_wav_data()) 
                # Using google to recognize audio
                #print("Recognising...")
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("You: ",MyText)
                #SpeakText(MyText)
                a=0
                return MyText
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("Could not identify voice")

#==========================================Predicting emotion part====================================#

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
            if i == total_secs - sec_per_split:
                #print('All splited successfully') 
                pass

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
def predict(model2):
    folder = "C:\\Users\\Ashwin\\Desktop\\code\\Voice_ML\\voice"                   #folder where the splitted audio will be there
    file = "speech.wav"                                         #audio file. Must be in same folder as above
    split_wav = SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(sec_per_split=4)

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

    #predicting...
    y_predict = model2.predict(x,verbose=0)

    #Getting proper output from y_predict    
    encoded_list=['angry','calm','disgust','fear','happy','neutral','sad','surprise']
    for i in range (0,length,1):
        index = np.argmin(np.abs(y_predict[i]-1.0000000))
        emotion=encoded_list[index]
        emotion_list.append(emotion)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

from torch.utils.data import Dataset, DataLoader

import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Hriday bot"
print("Let's chat! (Say 'quit' to exit)")
while True:
    empty()
    sentence = voice_recog()
    t1=threading.Thread(target=predict,name='t1',args=(model2,))
    t1.start()
    if sentence == "stop execution":
        print(f"{bot_name}: See you later!")
        SpeakText("See you later!")
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response=random.choice(intent['responses'])
                print(f"{bot_name}: ",response)
                SpeakText(response)
    else:
        print(f"{bot_name}: I do not understand...")
        SpeakText("I do not understand...")
    t1.join()
print(emotion_list)