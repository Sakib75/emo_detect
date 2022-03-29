#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
import soundfile
warnings.filterwarnings("ignore")


# ## Load the dataset

# In[30]:


from merge_datasets import merge_datasets

all_datasets = merge_datasets(bangla=True,english_tess=False)
all_datasets


# In[31]:



df = all_datasets[['speech','label']]
# df = df[df['label'] != 'disgust']

sns.countplot(df['label'])
df


# In[32]:


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion,size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def spectogram(data,sr,emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(10,4))
    plt.title(emotion,size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time',y_axis='hz')
    plt.colorbar()
    


# ## Sample sound

# In[33]:


emotions = ['neutral','angry']

for emotion in emotions:
    path = df['speech'][df['label'] == emotion]
    data, sampling_rate=librosa.load(path.iloc[1])
    waveplot(data, sampling_rate,emotion)
    spectogram(data,sampling_rate,emotion)
Audio(path.iloc[1])


# In[34]:


# audio_path1 = "./SUBESCO/SUBESCO/M_02_NIPUN_S_7_ANGRY_1.wav"
# audio_path2 = "./SUBESCO/SUBESCO/M_02_NIPUN_S_7_SAD_1.wav"
# audio_path3 = "./SUBESCO/SUBESCO/M_02_NIPUN_S_7_NEUTRAL_1.wav"
# audio_path4 = "./SUBESCO/SUBESCO/M_02_NIPUN_S_7_FEAR_1.wav"
# audio_paths = [audio_path1,audio_path2,audio_path3,audio_path4]
# for audio_path in audio_paths:
#     y, sr = librosa.load(audio_path)
#     zcr = librosa.feature.zero_crossing_rate(y).max()
#     print(zcr)
# sr


# ## Feature Extraction

# In[35]:


def extract_mfcc(filename):
    y ,sr = librosa.load(filename, duration=5, offset=0.5)
    mfcc =np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc


    
def extract_features(file_name, mfcc, chroma, mel, cent, zcr):
#     waveplot(data, sampling_rate,emotion)
#     spectogram(data,sampling_rate,emotion)
    y ,sr = librosa.load(file_name, duration=5, offset=0.5)
    result=np.array([])
    if(mfcc):
        mfccs =np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
#         print(mfccs.shape)
        result=np.hstack((result, mfccs))
    if(chroma):
        stft=np.abs(librosa.stft(y))
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
#         print(chroma.shape)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y, sr=sr,win_length=1000).T,axis=0)
#         print(mel.shape)
        result=np.hstack((result, mel))
    if cent:
        cent=np.mean(librosa.feature.tonnetz(y, sr=sr).T,axis=0)
#         print(cent.shape)
        result=np.hstack((result, cent))
    if zcr:
        zcr =np.mean(librosa.feature.zero_crossing_rate(y=y,frame_length=5, hop_length  = 5100 ).T,axis=1)
#         print(zcr.shape)
        result=np.hstack((result, zcr))
    return result

# extract_features("./SUBESCO/SUBESCO/M_02_NIPUN_S_7_HAPPY_4.wav", mfcc=True, chroma=True, mel=True)


# ## Extract mfcc 

# In[501]:


X_mfcc = df['speech'].apply(lambda x: extract_features(x, mfcc=True, chroma=True, mel=False,cent=False,zcr=False))


# In[36]:


type(X_mfcc)


# ## Expand dimension

# In[ ]:





# In[ ]:





# In[8]:


X = [ x for x in X_mfcc]

X= np.array(X)  
type(X)
# np.save('./datasets_feature_extracted/banglaenglish__mfcc_chroma.npy', X)
# np.save('./datasets_feature_extracted/bangla__mfcc_chroma.npy', X)
np.save('./datasets_feature_extracted/english__mfcc_chroma.npy', X)


# In[37]:



# new_num_arr = np.load('./datasets_feature_extracted/banglaenglish__mfcc_chroma.npy')
# new_num_arr = np.load('./datasets_feature_extracted/bangla__mfcc_chroma.npy')
new_num_arr = np.load('./datasets_feature_extracted/bangla__mfcc_chroma.npy')

X = new_num_arr
X


# In[38]:



X=np.expand_dims(X, -1)
print(X.shape)
print(X[0])


# In[39]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

y = enc.fit_transform(df[['label']])

print(y.shape)
y
enc.categories_[0]


# In[40]:


y = y.toarray()
print(y.shape)


# In[41]:


inp_shape = X[0].shape[0]
inp_shape

input_shape
# ## LSTM MODEL

# In[15]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Conv1D,MaxPooling1D,Flatten,GlobalAveragePooling1D,BatchNormalization
from keras.callbacks import ReduceLROnPlateau


# In[ ]:





# In[588]:


unq = len(df['label'].unique())
# model = Sequential([
#     LSTM(128, return_sequences=True, input_shape=(inp_shape,1)),
#     LSTM(64, return_sequences=False, input_shape=(inp_shape,1)),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(unq, activation='softmax'),
# ])

# model = Sequential([
#     Conv1D(128, 2, activation="relu", input_shape=(inp_shape,1)),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2,strides=2, padding='valid'),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2,strides=2, padding='valid'),
#     Dropout(0.2),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     BatchNormalization(),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2,strides=2, padding='valid'),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     MaxPooling1D(pool_size=1,strides=2, padding='valid'),
#     GlobalAveragePooling1D(),
#     Dropout(0.2),
#     Dense(unq, activation='softmax'),
# ])

# cv + lstm
model = Sequential([
    Conv1D(128, 2, activation="relu", input_shape=(inp_shape,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2,strides=2, padding='valid'),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2,strides=2, padding='valid'),
    Dropout(0.2),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    BatchNormalization(),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2,strides=2, padding='valid'),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
    MaxPooling1D(pool_size=1,strides=2, padding='valid'),
#     GlobalAveragePooling1D(),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(unq, activation='softmax'),
])

# model = Sequential([
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     MaxPooling1D(pool_size=2,strides=2, padding='valid'),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     MaxPooling1D(pool_size=2,strides=2, padding='valid'),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     Conv1D(64, 2, activation="relu", input_shape=(inp_shape,1)),
#     MaxPooling1D(),
# #     GlobalAveragePooling1D(),
#     LSTM(128),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(16, activation='relu'),
#     Dropout(0.2),
#     Dense(unq, activation='softmax'),
# ])



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[567]:


import tensorflow as tf
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5,
    min_lr=0.00001,

)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,

    restore_best_weights=True,
)


# ## Train the model

# In[42]:


# y = np.asarray(y).astype(np.float32)
# X = np.asarray(X).astype(np.float32)
# print(y)


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle= True)
# history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=100, validation_data=(x_valid, y_valid), callbacks=[reduce_lr, early_stop])


# ## Plot the result

# In[43]:


Y_pred = model.predict(X)

np.argmax(Y_pred)

y_pred_labels = [np.argmax(i) for i in Y_pred]
y_real_labels = [np.argmax(i) for i in y]


# In[570]:


import tensorflow as tf
cm = tf.math.confusion_matrix(labels=y_real_labels,predictions=y_pred_labels)


# In[571]:


import seaborn as sn
plt.figure(figsize= (10,7))
sn.heatmap(cm,annot=True,fmt='d',xticklabels=enc.categories_[0],yticklabels=enc.categories_[0])
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[572]:


import pickle


# In[573]:



model.save("./models/en_99_lstm.h5")


# In[61]:


from tensorflow import keras
loaded_model = keras.models.load_model('./models/bnen_78_lstm.h5')
loaded_model.summary()


# In[62]:


Y_pred = loaded_model.predict(x_valid)


# In[63]:




y_pred_labels = [np.argmax(i) for i in Y_pred]
y_real_labels = [np.argmax(i) for i in y_valid]
import tensorflow as tf
cm = tf.math.confusion_matrix(labels=y_real_labels,predictions=y_pred_labels)
import seaborn as sn
plt.figure(figsize= (10,7))
sn.heatmap(cm,annot=True,fmt='d',xticklabels=enc.categories_[0],yticklabels=enc.categories_[0])
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[64]:


loaded_model.evaluate(x_valid, y_valid)


# In[577]:


epochs = list(range(29))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[578]:


epochs = list(range(29))
acc = history.history['loss']
val_acc = history.history['val_loss']

plt.plot(epochs, acc, label='train loss')
plt.plot(epochs, val_acc, label='val loss')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[494]:


from tensorflow import keras
loaded_model = keras.models.load_model('./models/english_3_cv_lstm.h5')
loaded_model.summary()
from keras.utils.vis_utils import plot_model
plot_model(loaded_model, to_file='english_3_cv_lstm.png', show_shapes=True, show_layer_names=True)


# In[2]:


import keras.utils.vis_utils
from importlib import reload
reload(keras.utils.vis_utils)


from keras.utils.vis_utils import plot_model    
plot_model(loaded_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[589]:


from tensorflow.keras.utils import plot_model
loaded_model = keras.models.load_model('./models/english_3_cv_lstm.h5')
# model = Model(...)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[32]:


for i in range (0,3) :
    j = ['pogo', 'mati', 'syed+A']
    print("Shakib" +" loves " + j[i])


# In[ ]:




