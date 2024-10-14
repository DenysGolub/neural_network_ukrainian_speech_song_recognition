import joblib
import sklearn
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import librosa
from tkinter import *
from tkinter import messagebox

def chose_songs():
    file_paths = filedialog.askopenfilenames()
    results = []
    for file_path in file_paths:
        # tp = get_spectral_features(file_path)
        # lang = predict(tp)
        # results.append(f"{os.path.basename(file_path)}: {lang}")
        listbox.insert(END, file_path)
    # Update the listbox with the results
    
def get_song_name(file_path):
        # Split the string by the slash '/'
    parts = file_path.split('/')
    
    # Return the last part of the list
    return parts[-1]
    
    
def get_spectral_features(file_path):
    scaler = joblib.load('scaler_train.save')
    y_new, sr_new = librosa.load(file_path, mono=True, duration=30)

    chroma_stft_new = librosa.feature.chroma_stft(y=y_new, sr=sr_new)
    rmse_new = librosa.feature.rms(y=y_new)[0]
    spec_cent_new = librosa.feature.spectral_centroid(y=y_new, sr=sr_new)
    spec_bw_new = librosa.feature.spectral_bandwidth(y=y_new, sr=sr_new)
    rolloff_new = librosa.feature.spectral_rolloff(y=y_new, sr=sr_new)
    zcr_new = librosa.feature.zero_crossing_rate(y_new)
    mfcc_new = librosa.feature.mfcc(y=y_new, sr=sr_new)

    to_predict = np.mean(chroma_stft_new), np.mean(rmse_new), np.mean(spec_cent_new), np.mean(spec_bw_new), np.mean(rolloff_new), np.mean(zcr_new)
    for e in mfcc_new:
        to_predict += (np.mean(e),)
    to_predict = scaler.transform([to_predict])
    
    return to_predict

def predict(to_predict):
    encoder = joblib.load('encoder_train.save')
    encoder = LabelEncoder()
    types = ['ukrainian', 'other']
    model = keras.models.load_model('model.keras')
    prediction = model.predict(to_predict)
    predicted_genre_index = np.argmax(prediction)

    encoder.fit(types)
    predicted_lang = encoder.inverse_transform([predicted_genre_index])[0]
    
    return predicted_lang

def list_box_pred(value):
    selection = listbox.curselection()
    if selection:
        index = selection[0]
        # Get the selected item text
        item_text = listbox.get(index)
        lang = predict(get_spectral_features(item_text))
        messagebox.showinfo("Аналіз", lang)
        
root = Tk()
root.title('App')
root.geometry('520x300')

# Create the listbox to display results
listbox = Listbox(root, width=83, height=15)
listbox.bind("<Double-1>", list_box_pred)
listbox.grid(column=0, row=0, padx=10, pady=10, columnspan=2)

button = tk.Button(text="Обрати пісні", command=chose_songs)
button.grid(column=0, row=1, padx=200, pady=0)

root.mainloop()
