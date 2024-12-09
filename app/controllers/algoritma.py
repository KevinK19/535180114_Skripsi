from flask import render_template, flash, redirect, url_for, request
from app.models.Dataset import *
from app.models.Detail import *
from app.models.Evaluasi import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
tf.config.experimental_run_functions_eagerly(True)

from tqdm import tqdm
import time

def index():
    np.random.seed(123)
    tf.random.set_seed(123)
    tf.keras.utils.set_random_seed(123)

    list_split = ['60% training dan 40% testing', '70% training dan 30% testing', '80% training dan 20% testing', '90% training dan 10% testing']
    df_split = pd.DataFrame({'keterangan': list_split, 'test_split': [0.4, 0.3, 0.2, 0.1]})

    list_data = Dataset.get().serialize()
    show_data = 0
    nama_data = ''
    test_split = ''
    df_train  = ''
    df_test   = ''
    evaluasi  = ''
    labels = ''
    value1 = ''
    value2 = ''
    value3 = ''
    if len(request.args) > 0:
        show_data = 1
        nama_data = request.args['namaData']
        test_split = float(request.args['splitdata'])

        dataset = Dataset.where('nama_data', nama_data).first().serialize()
        detailData = Detail.where('dataset_id', dataset['id']).get().serialize()

        df = pd.DataFrame(detailData)
        df = df.drop(columns=['id', 'dataset_id', 'created_at', 'updated_at', 'deleted_at'])
        df = df.dropna()
        # df = df.tail(35)

        # Ubah kolom Date menjadi datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        # Memisahkan Tahun, Bulan, dan Hari
        # df['Year']  = df['date'].dt.year
        # df['Month'] = df['date'].dt.month
        # df['Day']  = df['date'].dt.day
        # print(df)

        df_select = df[['open', 'close']]
        # Menghitung rata-rata
        df_select['average'] = df_select.mean(axis=1)
        # print(df_select)

        # --------------- Z-Score Normalization ---------------
        # Inisialisasi StandardScaler
        scaler = StandardScaler()

        # Fit dan transform data
        df_scaled = scaler.fit_transform(df_select['average'].values.reshape(-1, 1))
        # Mengubah hasil kembali ke DataFrame
        df_scaled = pd.DataFrame({'average': df_scaled.flatten()})
        print(df_scaled)

        # --------------- Split Data --------------
        X = df_scaled['average']
        y = df_scaled['average']

        # Membagi data menjadi training dan testing
        split_size = int(len(df) * (1-test_split))

        X_train, X_test, y_train, y_test = make_train_test_splits(X, y, split_size)

        print('Jumlah data training :', len(X_train))
        print('Jumlah data testing  :', len(X_test))

        # --------------- LSTM ---------------
        LSTMmodel = Sequential()
        LSTMmodel.add(LSTM(units=100, return_sequences=True, input_shape=(1, 1)))
        LSTMmodel.add(Dropout(0.3))
        LSTMmodel.add(LSTM(units=100, return_sequences=True))
        LSTMmodel.add(Dropout(0.3))
        LSTMmodel.add(LSTM(units=100))
        LSTMmodel.add(Dropout(0.3))
        LSTMmodel.add(Dense(units=1))
        LSTMmodel.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')


        BATCH_SIZE = 32

        callbacks_lstm = [ModelCheckpoint('static/best_model_LSTM.h5', monitor='val_loss', verbose=1, save_best_only=True),
                            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                            TqdmCallback(total_epochs=100, algorithm_name="LSTM")]

        history_lstm = LSTMmodel.fit(X_train, 
                                    y_train, 
                                    epochs=100,
                                    batch_size=BATCH_SIZE,
                                    callbacks=callbacks_lstm,
                                    validation_data=(X_test, y_test))
        
        n_model_lstm = load_model('static/best_model_LSTM.h5')

        # Prediksi
        y_pred_lstm = n_model_lstm.predict(X_test)

        _mape_lstm = mape(scaler.inverse_transform(y_test.values.reshape(1, -1)), scaler.inverse_transform(y_pred_lstm.reshape(1, -1)))
        _mae_lstm  = mean_squared_error(scaler.inverse_transform(y_test.values.reshape(1, -1)), scaler.inverse_transform(y_pred_lstm.reshape(1, -1)))
        print(f'MAPE : {_mape_lstm}')
        print(f'MAE : {_mae_lstm}')

        # --------------- BiLSTM ---------------
        BiLSTMmodel = Sequential()
        BiLSTMmodel.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(1, 1)))
        BiLSTMmodel.add(Dropout(0.3))
        BiLSTMmodel.add(Bidirectional(LSTM(units=100, return_sequences=True)))
        BiLSTMmodel.add(Dropout(0.3))
        BiLSTMmodel.add(Bidirectional(LSTM(units=100)))
        BiLSTMmodel.add(Dropout(0.3))
        BiLSTMmodel.add(Dense(units=1))
        BiLSTMmodel.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

        callbacks_bilstm = [ModelCheckpoint('static/best_model_BiLSTM.h5', monitor='val_loss', verbose=1, save_best_only=True),
                            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                            TqdmCallback(total_epochs=100, algorithm_name="BiLSTM")]

        history_bilstm = BiLSTMmodel.fit(X_train, 
                                        y_train,    
                                        epochs=100,
                                        batch_size=BATCH_SIZE,
                                        callbacks=callbacks_bilstm,
                                        validation_data=(X_test, y_test))
        
        n_model_bilstm = load_model('static/best_model_BiLSTM.h5')

        # Prediksi
        y_pred_bilstm = n_model_bilstm.predict(X_test)

        _mape_bilstm = mape(scaler.inverse_transform(y_test.values.reshape(1, -1)), scaler.inverse_transform(y_pred_bilstm.reshape(1, -1)))
        _mae_bilstm  = mean_squared_error(scaler.inverse_transform(y_test.values.reshape(1, -1)), scaler.inverse_transform(y_pred_bilstm.reshape(1, -1)))
        print(f'MAPE : {_mape_bilstm}')
        print(f'MAE : {_mae_bilstm}')

        df_train = pd.DataFrame(X_train)
        df_train['Date']  = df['date'][:split_size]
        # Mengambil hanya bagian tanggal
        df_train['Date'] = df_train['Date'].dt.date
        df_train['Close'] = scaler.inverse_transform(y_train.values.reshape(1, -1)).flatten()
        print(df_train)

        df_test = pd.DataFrame(X_test)
        df_test['Date']  = df['date'][split_size:]
        # Mengambil hanya bagian tanggal
        df_test['Date']   = df_test['Date'].dt.date
        df_test['Close']  = scaler.inverse_transform(y_test.values.reshape(1, -1)).flatten()
        df_test['LSTM']   = scaler.inverse_transform(y_pred_lstm.reshape(1, -1)).flatten()
        df_test['BiLSTM'] = scaler.inverse_transform(y_pred_bilstm.reshape(1, -1)).flatten()
        print(df_test)

        labels = [label.strftime('%Y-%m-%d') for label in df_test['Date'].values]
        value1 = df_test['Close'].values.tolist()
        value2 = df_test['LSTM'].values.tolist()
        value3 = df_test['BiLSTM'].values.tolist()

        evaluasi = pd.DataFrame({
            'algoritma': ['LSTM', 'Bilstm'],
            'MAPE(%)'  : [_mape_lstm, _mape_bilstm],
            'MAE'      : [_mae_lstm, _mae_bilstm],
        })

        # Mengecek apakah ada data dengan dataset_id yang sama
        existing_evaluasi = Evaluasi.where('dataset_id', dataset['id']).where('test_split', test_split).first()

        if existing_evaluasi is not None:
            # Jika data sudah ada, update
            existing_evaluasi.mape_lstm = _mape_lstm
            existing_evaluasi.mae_lstm = round(_mae_lstm, 10)
            existing_evaluasi.mape_bilstm = _mape_bilstm
            existing_evaluasi.mae_bilstm = round(_mae_bilstm, 105)
            existing_evaluasi.save()
        else:
            # Jika data tidak ada, buat baru
            save_evaluasi = Evaluasi()
            save_evaluasi.dataset_id = dataset['id']
            save_evaluasi.test_split = test_split
            save_evaluasi.mape_lstm = _mape_lstm
            save_evaluasi.mae_lstm = round(_mae_lstm, 10)
            save_evaluasi.mape_bilstm = _mape_bilstm
            save_evaluasi.mae_bilstm = round(_mae_bilstm, 10)
            save_evaluasi.save()

    return render_template('pages/algoritma.html', segment='algoritma', df_split=df_split, list_data=list_data, test_split=test_split,
    nama_data=nama_data, df_train=df_train, df_test=df_test, show_data=show_data, evaluasi=evaluasi, 
    labels=labels, value1=value1, value2=value2, value3=value3)

# Membuat fungsi untuk membagi data menjadi training dan testing
def make_train_test_splits(X, y, split_size):
    X_train = X[:split_size]
    y_train = y[:split_size]
    X_test  = X[split_size:]
    y_test  = y[split_size:]
    return X_train, X_test, y_train, y_test

def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return round(mape*100, 2)

class TqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, algorithm_name):
        super().__init__()
        self.total_epochs = total_epochs
        self.algorithm_name = algorithm_name
        self.epoch_bar = tqdm(total=total_epochs, desc=f'Training {algorithm_name}', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        # Update progress bar
        self.epoch_bar.update(1)

        # Simpan progress ke file
        with open("static/progress.txt", "w") as f:
            f.write(f"{self.algorithm_name}: Epoch {epoch + 1}/{self.total_epochs}\n")

    def on_train_end(self, logs=None):
        self.epoch_bar.update(self.total_epochs - self.epoch_bar.n)  # Make sure to fill the progress bar if needed
        self.epoch_bar.close()  # Properly close the progress bar

        # Ensure progress is stopped in the text file
        with open("static/progress.txt", "a") as f:
            f.write(f"{self.algorithm_name}: Training completed\n")

        # Clear the file after training
        with open("static/progress.txt", "w") as f:
            f.write("")  # Overwrite the file with an empty string
