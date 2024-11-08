
# imports
import librosa
import os
import glob
import wave
import numpy as np
import pickle
import sys
from numpy import array
from numpy import asarray
from numpy import zeros
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import keras
from keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Dense, Bidirectional, LSTM
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Reshape, LeakyReLU
from keras.layers import UpSampling2D, UpSampling1D
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.callbacks import ModelCheckpoint
import IPython.display as ipd
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

# Connecting to google drive storage
from google.colab import drive
drive.mount('/content/gdrive')

# paths
path = "/content/gdrive/My Drive/Dataset/Test/wavs"
spectrogram_path = "/content/gdrive/My Drive/Dataset/Test/Spectrogram/"
Mel_spectrogram_path = "/content/gdrive/My Drive/Dataset/Test/Mel-Spectrogram/"
text_path = '/content/gdrive/My Drive/Dataset/Test/metadata.csv'

# Dataset Preprocessing

orig = []
hop = []
i = 0
for filename in glob.glob(os.path.join(path, '*.wav')):
  scale, sr = librosa.load(filename)
  # print(scale.shape[0], sr)
  mel_spectrogram = librosa.feature.melspectrogram(scale, sr = sr, n_fft = 2048, hop_length = int(scale.shape[0] / 255), n_mels = 128)
  hop.append((filename, scale.shape[0]/255))

  log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

  orig.append(np.transpose(np.array([mel_spectrogram])))
  i += 1 
  if(i == 10):
    break
    print(i)

orig = np.array(orig)
print('orig: ', orig.shape)
print('hop: ', len(hop))
# saving
np.save('/content/gdrive/My Drive/Dataset/Test/mel_proper', orig)

import pickle
with open('/content/gdrive/My Drive/Dataset/Test/hop_length.pkl', 'wb') as fp:
  pickle.dump(hop, fp)

ret = np.transpose(orig[5])
print(ret[0].shape)

# preprocessing data

import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd

def read_audio_from_filename(filename):
    audio, sr = librosa.load(filename)
    print(audio.shape, int(audio.shape[0]/256))
    # D = np.abs(librosa.stft(audio))**2
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft = 2048, hop_length = int(audio.shape[0]/256), n_mels = 128)
    # crop = cv2.resize(mel_spectrogram,(256, 128))
    return mel_spectrogram
def convert_data():
    wav_filename = "/content/gdrive/My Drive/Dataset/Test/wavs/LJ001-0001.wav"
    audio = read_audio_from_filename(wav_filename)
    print(audio.shape, type(audio))
    return audio
specto = convert_data()
print('sd')
res = librosa.feature.inverse.mel_to_audio(specto)
print(res.shape, type(res)) 
import soundfile as sf
sf.write("/content/gdrive/My Drive/Dataset/Test/hop_file1.wav", res, 22050)
res1 = librosa.feature.inverse.mel_to_audio(specto, hop_length=831)
sf.write("/content/gdrive/My Drive/Dataset/Test/hop_file2.wav", res1, 22050)

"""# Preprocessed Data Cleaning"""

# test-check cell
import cv2
from PIL import Image
import pickle

with open('/content/gdrive/My Drive/Dataset/Test/hop_length_1.pkl', 'rb') as fp:
  l = pickle.load(fp)

print(len(l), l[:20])
p = []
for i in range(len(l)):
  if(len(l[i][0]) != 57):
    print(l[i], i)
  else:
    p.append((l[i][0], l[i][1], i))
p = sorted(p, key=lambda x:x[0])
print(p)
orig = np.load('/content/gdrive/My Drive/Dataset/Test/mel_proper_1.npy', allow_pickle=True)
temp = []
for i in orig:
  if(i.shape == (256, 128, 1)):
    temp.append(i)
temp = np.array(temp)
print(temp.shape)
np.save('/content/gdrive/My Drive/Dataset/Test/mel_proper_256', temp)
audio_mel = np.load('/content/gdrive/My Drive/Dataset/Test/mel_proper_1.npy', allow_pickle=True)
temp = np.load('/content/gdrive/My Drive/Dataset/Test/mel_proper_256.npy')
print(audio_mel.shape)

res1 = librosa.feature.inverse.mel_to_audio(specto, hop_length=831)
sf.write("/content/gdrive/My Drive/Dataset/Test/hop_file2.wav", res1, 22050)

print(np.array(l).shape)
t = Tokenizer()
t.fit_on_texts(docs) 
vocab_size = len(t.word_index) + 1
print('vocab size ', vocab_size)
print(t.word_index)
p = np.load('/content/gdrive/My Drive/Dataset/Test/mel_test_5.npy')
print(p.shape)
embeddings_index = {}
f = open('/content/gdrive/My Drive/Dataset/Test/Glove/glove.6B.100d.txt')
i = 0
for line in f:
  values = line.split()
  word = values[0]
  coefs = asarray(values[1:], dtype = 'float32')
  print(coefs.shape)
  embeddings_index[word] = coefs
  i += 1
  print(values)
  print(embeddings_index)
  if(i > 2):
    break
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 256))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
print(embedding_vector)

final_audio = []
final_hop = []
print(audio_mel[7099].shape)

index_to_remove_258 = [1913, 3726, 4660, 4703, 6431, 7099, 7386, 8130, 12398]
index_to_remove_extra = [2363, 4607, 6166, 6649, 6956, 10016, 10748, 12010, 12693]
index_to_remove_text = []
for i in p:
  if(i[2] in index_to_remove_258):
    print(i)
    index_to_remove_text.append(i[0][43:53])
  else:
    final_hop.append(i)
print(len(final_hop))
print(index_to_remove_text)

for i in range(len(audio_mel)):
  if(audio_mel[i].shape != (256, 128, 1) or i in index_to_remove_extra):
    print(i)
  else:
    final_audio.append(audio_mel[i])

final_audio = np.array(final_audio)
print(final_audio.shape)


# test-check cell
import cv2
from PIL import Image
# temp = Image.open('/content/gdrive/My Drive/Dataset/Test/Mel-Spectrogram/LJ001-0004.png')
# print(np.array(temp).shape)
# temp = Image.open('/content/gdrive/My Drive/Dataset/Test/Mel-Spectrogram/LJ001-0012.png')
# print(np.array(temp).shape)
# temp = Image.open('/content/gdrive/My Drive/Dataset/Test/Mel-Spectrogram/LJ001-0047.png')
# print(np.array(temp).shape)
import pickle

with open('/content/gdrive/My Drive/Dataset/Test/hop_length_1.pkl', 'rb') as fp:
  l = pickle.load(fp)

print(len(l), l[:20])
p = []
for i in range(len(l)):
  if(len(l[i][0]) != 57):
    print(l[i], i)
  else:
    p.append((l[i][0], l[i][1], i))
p = sorted(p, key=lambda x:x[0])
print(p)
# orig = np.load('/content/gdrive/My Drive/Dataset/Test/mel_proper_1.npy', allow_pickle=True)
# temp = []
# for i in orig:
#   if(i.shape == (256, 128, 1)):
#     temp.append(i)
# temp = np.array(temp)
# print(temp.shape)
# np.save('/content/gdrive/My Drive/Dataset/Test/mel_proper_256', temp)
audio_mel = np.load('/content/gdrive/My Drive/Dataset/Test/mel_proper_1.npy', allow_pickle=True)
temp = np.load('/content/gdrive/My Drive/Dataset/Test/mel_proper_256.npy')
print(audio_mel.shape)

res1 = librosa.feature.inverse.mel_to_audio(specto, hop_length=831)
sf.write("/content/gdrive/My Drive/Dataset/Test/hop_file2.wav", res1, 22050)

print(np.array(l).shape)
t = Tokenizer()
t.fit_on_texts(docs) 
vocab_size = len(t.word_index) + 1
print('vocab size ', vocab_size)
print(t.word_index)
p = np.load('/content/gdrive/My Drive/Dataset/Test/mel_test_5.npy')
print(p.shape)
embeddings_index = {}
f = open('/content/gdrive/My Drive/Dataset/Test/Glove/glove.6B.100d.txt')
i = 0
for line in f:
  values = line.split()
  word = values[0]
  coefs = asarray(values[1:], dtype = 'float32')
  print(coefs.shape)
  embeddings_index[word] = coefs
  i += 1
  print(values)
  print(embeddings_index)
  if(i > 2):
    break
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 256))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
print(embedding_vector)

"""## Dataset Metadata Processing 

"""

# Dataset metadata processing
max_length = 0
docs = []
index = 0

with open(text_path, encoding = 'utf-8') as fp:
  for line in fp:
    parts = line.strip().split('|')
    # print(parts)
    if(parts[0] in index_to_remove_text):
      # print(parts[2])
      pass
    else:
      max_length = max(max_length, len(parts[2]))
      docs.append(parts[2])
      index += 1
    
print('maximum length of charcters in a sentence of training data is', max_length)
print('total length ', len(docs))
print(docs[:5])

# saving files (do not re-run)
print(final_audio.shape)
print(len(final_hop))
print(len(docs))

with open('/content/gdrive/My Drive/Dataset/Test/final_hop_length.pkl', 'wb') as fp:
  pickle.dump(final_hop, fp)

with open('/content/gdrive/My Drive/Dataset/Test/final_text.pkl', 'wb') as fp:
  pickle.dump(docs, fp)

np.save('/content/gdrive/My Drive/Dataset/Test/final_mel', final_audio)

import os
from pathlib import Path
path = "/content/gdrive/My Drive/Dataset/LibriSpeech/dev-clean/"

audio_data = {}
audio_path = []
c = 0
for root, dirs, files in os.walk(path):
  for file in files:
    if(file.endswith(".txt")):
      # print(os.path.join(root,file))
      name = os.path.join(root,file)
      #print(name)
      data =open(name,"r")
      # print(data.readline())
      for x in data:
        # print(x)
        
        path = str(os.path.join(root,file))
        # print(path)
        # if(str(x[0:14]) not in audio_data ):
          # print("okay")
        c+=1
        space = x[0]
        index = 0
        audio_name = ''
        while(space != ' '):
          audio_name += x[index]
          space = x[index]
          index+=1
        
        
        audio_data[str(x[0:index-1])] = x[index:]
    elif(file.endswith(".flac")):
      audio_path.append(str(os.path.join(root,file)))
      # print(Path(os.path.join(root,file)).stem)
print(len(audio_data))
print((audio_path))
print(c)
print(audio_data)


"""# Model Development"""

# part-1

# preparing tokenizer

docs = []
with open('/content/gdrive/My Drive/Dataset/Test/final_text.pkl', 'rb') as fp:
  docs = pickle.load(fp)

max_length = 187
t = Tokenizer()
t.fit_on_texts(docs) 
vocab_size = len(t.word_index) + 1
print('vocab size ', vocab_size)

encoded_docs = t.texts_to_sequences(docs)
print('encoded docs: ', encoded_docs)

# padding the documents to a max words containing sentence
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
# padded_docs = docs
print('padded docs: ', padded_docs)

# loading the whole embedding into memory
embeddings_index = dict()
f = open('/content/gdrive/My Drive/Dataset/Test/Glove/glove.6B.300d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype = 'float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

# For printing the stats of the numpy vector

def stats(l, s):
  print('-----------------------------')
  print('stats for: ', s)
  print('min: ', np.min(l))
  print('max: ', np.max(l))
  print('mean: ', np.mean(l))
  print('median: ', np.median(l))
  print('std dev: ', np.std(l))
  print('-----------------------------')

# Loading the saved content

input_1 = np.load('/content/gdrive/My Drive/Dataset/Test/final_padded.npy')
input_2 = np.load('/content/gdrive/My Drive/Dataset/Test/final_log_mel.npy')

output = np.load('/content/gdrive/My Drive/Dataset/Test/final_log_mel.npy')
print(input_1.shape)
print(input_2.shape)

stats(input_2, 'input_2')
stats(input_1, 'input_1')

# normalising input

inp_min = np.min(input_1)
inp_max = np.max(input_1)
inp_mean = np.mean(input_1)
inp_std = np.std(input_1)

scaled_data = (input_1 - inp_min) / (inp_max - inp_min)
scaled_data *= 1000
stats(scaled_data, 'input_1')
print(scaled_data.shape)
# input_1 = scaled_data

inp_min = np.min(input_2)
inp_max = np.max(input_2)
inp_mean = np.mean(input_2)
inp_std = np.std(input_2)

scaled_data = (input_2 - inp_min) / (inp_max - inp_min)
scaled_data *= 1000
stats(scaled_data, 'input_2')
print(scaled_data.shape)
# input_2 = scaled_data

# Input data figure plot
print(np.transpose(input_2[0])[0])
stats(np.transpose(input_2[0])[0], 'input_2')
stats(input_2[0], 'orig')
plt.figure(figsize=(12, 8))
plt.hist(np.transpose(input_2[0])[0], bins = 20, range = (-50,50))
plt.show()

# Preparing train, validation and test sets in the ratio 7:2:1
x_train_1 = input_1[:9100]
x_train_2 = input_2[:9100]
y_train = output[:9100]

x_val_1 = input_1[9100:11700]
x_val_2 = input_2[9100:11700]
y_val = output[9100:11700]

x_test_1 = input_1[11700:13000]
x_test_2 = input_2[11700:13000]
y_test = output[11700:13000]

# pipeline-1 -> Text Encoding
# model-1

model1 = Sequential()
model1.add(Embedding(vocab_size, 300,  weights=[embedding_matrix], input_length = 187, trainable=False))

model1.add(Convolution1D(filters=512, kernel_size=(5), activation='relu'))
model1.add(Convolution1D(filters=256, kernel_size=(5), activation='relu'))
model1.add(Convolution1D(filters=128, kernel_size=(3), activation='relu'))

model1.add(Bidirectional(LSTM(1024)))
model1.add(Reshape((64, 32, 1)))
model1.add(UpSampling2D())
model1.add(UpSampling2D())

model1.compile(loss='mse',optimizer='adam',metrics=['mse'])

print(model1.summary())

# pipeline-2 -> user/speaker encoding
# model-2

model2 = Sequential()

model2.add(Convolution2D(32, 5 ,data_format = 'channels_last', padding = 'same', activation = 'relu', strides = 1, input_shape = (256, 128, 1)))  
model2.add(AveragePooling2D(pool_size = (2, 2), padding = 'same'))  

model2.add(Convolution2D(64, 3, padding = 'same'))  
model2.add(AveragePooling2D(pool_size = (2, 2), padding = 'same'))  

layer_output = [layer.output for layer in model2.layers]
print(layer_output)

model2.add(Conv2DTranspose(24,(layer_output[3].get_shape().as_list()[1:3]),padding = 'same', activation = 'relu', strides = 2))
model2.add(Conv2DTranspose(1,(layer_output[0].get_shape().as_list()[1:3]),padding = 'same', activation = 'relu', strides = 2))

model2.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(model2.summary())

# merging two pipelines of models and preperation of final model

from keras.layers import *
from keras.models import Model

merged = Multiply()([model1.output, model2.output])

finalModel = Convolution2D(24, 5, padding = 'same', activation = 'relu', strides = 1)(merged)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel)

finalModel = Convolution2D(48, 3, padding = 'same')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Convolution2D(64, 3, padding = 'same')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Conv2DTranspose(48,(finalModel.shape[1:3]),padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(24,(finalModel.shape[1:3]),padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(1,(finalModel.shape[1:3]),padding = 'same', activation = 'relu', strides = 2)(finalModel)

combinedModel = Model([model1.input,model2.input], finalModel)
combinedModel.compile(optimizer='adam', loss='mse', metrics = [MeanAbsoluteError()])

print(combinedModel.summary())

combinedModel.fit([inp, output], output, epochs = 10)

# Trial 4
from keras.layers import *
from keras.models import Model, Input

model2 = Input(shape = (256, 128, 1))
merged = Multiply()([model1.output, model2])

finalModel = Convolution2D(64, 5, padding = 'same', activation = 'relu', strides = 1)(merged)
finalModel = MaxPooling2D(pool_size = (2, 2), padding = 'same')(finalModel)

finalModel = Convolution2D(128, 5, padding = 'same', activation = 'relu')(finalModel)
finalModel = MaxPooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Convolution2D(64, 3, padding = 'same', activation = 'relu')(finalModel)
finalModel = MaxPooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Conv2DTranspose(64, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(128, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(1, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)

combinedModel = Model([model1.input, model2], finalModel)

combinedModel.compile(optimizer='adam', loss='mse', metrics = ['mae', 'mse'])

print(combinedModel.summary())

filepath = "/content/gdrive/My Drive/Models/final_checkpoints/FT4.h5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

combinedModel.fit([x_train_1, k], y_train, epochs = 10, callbacks = callbacks_list)

combinedModel.save('/content/gdrive/My Drive/Models/final_trail/FT4.h5')

# merging two pipelines of models and preperation of final model

from keras.layers import *
from keras.models import Model, Input


model2 = Input(shape = (256, 128, 1))
merged = Multiply()([model1.output, model2])

finalModel = Convolution2D(24, 5, padding = 'same', activation = 'relu', strides = 1)(merged)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel)

finalModel = Convolution2D(48, 3, padding = 'same')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Convolution2D(64, 3, padding = 'same')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Conv2DTranspose(48, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(24, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(1, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)

combinedModel = Model([model1.input, model2], finalModel)

combinedModel.compile(optimizer='adam', loss='mse', metrics = [MeanAbsoluteError(), 'mse'])

print(combinedModel.summary())

filepath = "/content/gdrive/My Drive/Dataset/Test/model-checkpoints/model_2.h5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

combinedModel.fit([x_train_1, x_train_2], y_train, epochs = 10, validation_data = ([x_val_1, x_val_2], y_val), callbacks = callbacks_list)

combinedModel.save('/content/gdrive/My Drive/Models/Model-run-2.h5')

# merging two pipelines of models and preperation of final model
# trial_3
from keras.layers import *
from keras.models import Model, Input


model2 = Input(shape = (256, 128, 1))
merged = Multiply()([model1.output, model2])

finalModel = Convolution2D(24, 5, padding = 'same', activation = 'sigmoid', strides = 1)(merged)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel)

finalModel = Convolution2D(48, 3, padding = 'same', activation = 'sigmoid')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Convolution2D(64, 3, padding = 'same', activation = 'sigmoid')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Conv2DTranspose(48, (finalModel.shape[1:3]), padding = 'same', activation = 'sigmoid', strides = 2)(finalModel)
finalModel = Conv2DTranspose(24, (finalModel.shape[1:3]), padding = 'same', activation = 'sigmoid', strides = 2)(finalModel)
finalModel = Conv2DTranspose(1, (finalModel.shape[1:3]), padding = 'same', activation = 'sigmoid', strides = 2)(finalModel)

combinedModel = Model([model1.input, model2], finalModel)

combinedModel.compile(optimizer='adam', loss='mse', metrics = [MeanAbsoluteError(), 'mse'])

print(combinedModel.summary())

filepath = "/content/gdrive/My Drive/Dataset/Test/model-checkpoints/model_3.h5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

combinedModel.fit([x_train_1, x_train_2], y_train, epochs = 10, validation_data = ([x_val_1, x_val_2], y_val), callbacks = callbacks_list)

combinedModel.save('/content/gdrive/My Drive/Models/Model-run-3.h5')

# merging two pipelines of models and preperation of final model
# train_6
from keras.layers import *
from keras.models import Model, Input


model2 = Input(shape = (256, 128, 1))
merged = Multiply()([model1.output, model2])

finalModel = Convolution2D(24, 5, padding = 'same', activation = 'relu', strides = 1)(merged)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel)

finalModel = Convolution2D(48, 3, padding = 'same')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Convolution2D(64, 3, padding = 'same')(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Conv2DTranspose(48, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(24, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)
finalModel = Conv2DTranspose(1, (finalModel.shape[1:3]), padding = 'same', activation = 'relu', strides = 2)(finalModel)

combinedModel = Model([model1.input, model2], finalModel)

combinedModel.compile(optimizer='adam', loss='mae', metrics = [MeanAbsoluteError(), 'mse'])

print(combinedModel.summary())

filepath = "/content/gdrive/My Drive/Dataset/Test/model-checkpoints/model_4.h5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

combinedModel.fit([x_train_1, x_train_2], y_train, epochs = 10, validation_data = ([x_val_1, x_val_2], y_val), callbacks = callbacks_list)

combinedModel.save('/content/gdrive/My Drive/Models/Model-run-4.h5')

# merging two pipelines of models and preperation of final model
# train_5
from keras.layers import *
from keras.models import Model, Input


model2 = Input(shape = (256, 128, 1))
merged = Multiply()([model1.output, model2])

finalModel = Convolution2D(24, 5, padding = 'same',activation = LeakyReLU(alpha=0.03), strides = 1)(merged)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel)

finalModel = Convolution2D(48, 3, padding = 'same', activation = LeakyReLU(alpha=0.03))(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Convolution2D(64, 3, padding = 'same', activation = LeakyReLU(alpha=0.03))(finalModel)
finalModel = AveragePooling2D(pool_size = (2, 2), padding = 'same')(finalModel) 

finalModel = Conv2DTranspose(48, (finalModel.shape[1:3]), padding = 'same', activation = LeakyReLU(alpha=0.03), strides = 2)(finalModel)
finalModel = Conv2DTranspose(24, (finalModel.shape[1:3]), padding = 'same', activation = LeakyReLU(alpha=0.03), strides = 2)(finalModel)
finalModel = Conv2DTranspose(1, (finalModel.shape[1:3]), padding = 'same', activation = LeakyReLU(alpha=0.03), strides = 2)(finalModel)

combinedModel = Model([model1.input, model2], finalModel)

combinedModel.compile(optimizer='adam', loss='mae', metrics = [MeanAbsoluteError(), 'mse'])

print(combinedModel.summary())

filepath = "/content/gdrive/My Drive/Dataset/Test/model-checkpoints/model_5.h5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

combinedModel.fit([x_train_1, x_train_2], y_train, epochs = 10, validation_data = ([x_val_1, x_val_2], y_val), callbacks = callbacks_list)

combinedModel.save('/content/gdrive/My Drive/Models/Model-run-5.h5')


"""# Checks

"""
# test-check cell
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd

hops = []
with open('/content/gdrive/My Drive/Dataset/Test/final_hop_length.pkl', 'rb') as fp:
  hops = pickle.load(fp)

print(input_2.shape)
print(hops[0])
# inp_mean = np.median(input_2)


# logged = []

# for i in range(13091):
#   temp = librosa.power_to_db(np.transpose(input_2[i])[0])
#   logged.append(np.transpose(np.array([temp.tolist()])))

# lo = librosa.power_to_db(np.transpose(input_2[0])[0])
# print(lo.shape, np.min(lo), np.max(lo), np.mean(lo), np.median(lo))
# lo = (lo - np.min(lo))/(np.max(lo) - np.min(lo))
# print(lo.shape, np.min(lo), np.max(lo), np.mean(lo), np.median(lo))
# print(lo)

# lo = librosa.db_to_power(lo)
# print(lo.shape, np.min(lo), np.max(lo), np.mean(lo), np.median(lo))
# lo = (lo - np.mean(lo))/np.std(lo)
# print(lo.shape, np.min(lo), np.max(lo), np.mean(lo), np.median(lo))

# inp_std = np.std(input_2)
# normalized = (input_2 - inp_mean) / (inp_std)
# print(normalized)
# print(np.min(normalized), np.max(normalized))
print(np.max(op[0]), np.min(op[0]), np.median(op[0]), np.mean(op[0]))
# print(np.max(input_2*100), np.min(input_2*100), np.median(input_2*100), np.mean(input_2*100))
recovered = np.transpose(op[0])[0]
# recovered /= 100
print(recovered)

op_mean = np.mean(y_train[0])
op_std = np.std(y_train[0])

recovered = (recovered * (op_std)) + op_mean
print(recovered)
print(recovered, np.min(recovered), np.max(recovered), np.mean(recovered), np.median(recovered))
recovered = librosa.db_to_power(recovered)
print(recovered, np.min(recovered), np.max(recovered), np.mean(recovered), np.median(recovered))
res = librosa.feature.inverse.mel_to_audio(recovered, hop_length=835)
print(res.shape, type(res)) 
import soundfile as sf
sf.write("/content/gdrive/My Drive/Dataset/Test/FT1-1.wav", res, 22050)

"""# Inferencing

"""

def post_process(op):
  recovered = np.transpose(op)[0]
  recovered = librosa.db_to_power(recovered)
  res = librosa.feature.inverse.mel_to_audio(recovered, hop_length=835)
  filename = "/content/gdrive/My Drive/prediction-2.wav"
  sf.write(filename, res, 22050)
  return filename

# Loading the final saved model for prediction
saved_model = load_model('/content/gdrive/My Drive/Models/Final_Model.h5')

# Printing the summary of the final saved model
saved_model.summary()

# Reference audio used
ipd.Audio("/content/gdrive/My Drive/sample-1.wav")



# Input Example for demo
inp_text = input("Enter the text to be converted to audio: ")
reference_audio = x_test_2[0]

print('Processing...')

op = saved_model.predict([inp_text, reference_audio])

generated_audio = post_process(op)
print(generated_audio)
ipd.Audio(generated_audio)

# Initial Phase 1

import IPython.display as ipd
ipd.Audio("/content/gdrive/My Drive/Dataset/Test/FT1-1.wav")

# Initial Phase 2

import IPython.display as ipd
ipd.Audio("/content/gdrive/My Drive/Dataset/Test/test-2.wav")

