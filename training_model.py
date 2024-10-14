from keras.models import Sequential
import librosa
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras

def generate_dataset():
    # Generating a dataset
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Modify the label here
    types = ['ukrainian', 'other']

    for t in types:
        for filename in os.listdir(f'./music/{t}/'):
            songname = f'./music/{t}/{filename}'
            y, sr = librosa.load(songname, mono=True)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse=librosa.feature.rms(y=y)[0]
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            # Append the label to each row
            to_append += f' {t}'
            file = open('data.csv', 'a', newline='', encoding='UTF-8')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
    return data

def create_model(X_train):
    model = Sequential()
    
    model.add(keras.layers.Dense(27,activation='relu',input_shape=X_train[1].shape))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32,activation='relu')) 
    model.add(keras.layers.Dropout(0.3)) 
    
    model.add(keras.layers.Dense(2,activation='softmax'))
    return model

data = pd.read_csv('data.csv', encoding='latin-1', on_bad_lines='skip')
y=data.label
data = data.drop(['filename'], axis=1)
data = data.drop(['label'], axis=1)


# Label encoding for categoricals
for colname in data.select_dtypes("object"):
    data[colname], _ = data[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = data.dtypes == int

from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(data, y, discrete_features)
print(mi_scores)  # show a few features with their MI scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
plt.show()
# encoder = LabelEncoder()
# y = encoder.fit_transform(data['label'])
# scaler = StandardScaler()

# X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = create_model(X_train)

# model.compile(
#                optimizer=keras.optimizers.RMSprop(),
#                 loss=keras.losses.SparseCategoricalCrossentropy(),
#                 metrics=['accuracy'],
#             )

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=4, min_lr=0.0000000001)

# history = model.fit(X_train,
#                     y_train,
#                     epochs=500, 
#                     validation_data=(X_test, y_test),
#                     batch_size=32,
#                     callbacks=[early_stopping, reduce_lr]
#                     )

# test_loss, test_acc = model.evaluate(X_test, y_test)
# print('test_acc: ', test_acc)
# print('test_loss: ', test_loss)
# keras.models.save_model(filepath=os.getcwd()+'\model.keras', model=model)
# joblib.dump(scaler , 'scaler_train.save')
# joblib.dump(encoder, 'encoder_train.save')


# plt.plot(history.history['accuracy'], label='Training acc')
# plt.plot(history.history['val_accuracy'],  label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(history.history['loss'],  label='Training loss')
# plt.plot(history.history['val_loss'],  label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()