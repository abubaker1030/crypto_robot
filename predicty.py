# tensorflow_version 2.0
from flask import Flask, jsonify, request
import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
# %matplotlib inline
def predict_data(coin,limit):
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym={}&tsym=CAD&limit=1750'.format(coin))
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'

    hist.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)

    hist.tail(5)

    hist['high_low_avg'] = (hist['high'] + hist['low']) / 2
    hist = hist.drop(['high'], axis=1)
    hist = hist.drop(['low'], axis=1)
    hist.tail(5)

    def train_test_split(df, test_size=0.2):
        split_row = len(df) - int(test_size * len(df))
        train_data = df.iloc[:split_row]
        test_data = df.iloc[split_row:]
        return train_data, test_data

    train, test = train_test_split(hist, test_size=0.2)


    def normalise_zero_base(df):
        return df / df.iloc[0] - 1

    def extract_window_data(df, window_len=5, zero_base=True):
        window_data = []
        for idx in range(len(df) - window_len):
            tmp = df[idx: (idx + window_len)].copy()
            if zero_base:
                tmp = normalise_zero_base(tmp)
            window_data.append(tmp.values)
        return np.array(window_data)

    def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
        train_data, test_data = train_test_split(df, test_size=test_size)
        X_train = extract_window_data(train_data, window_len, zero_base)
        X_test = extract_window_data(test_data, window_len, zero_base)
        y_train = train_data[target_col][window_len:].values
        y_test = test_data[target_col][window_len:].values
        if zero_base:
            y_train = y_train / train_data[target_col][:-window_len].values - 1
            y_test = y_test / test_data[target_col][:-window_len].values - 1

        return train_data, test_data, X_train, X_test, y_train, y_test

    def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                        dropout=0.2, loss='mse', optimizer='adam'):
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))

        model.compile(loss=loss, optimizer=optimizer)
        return model

    np.random.seed(42)
    window_len = 5
    
    test_size = 0.2
    zero_base = True
    lstm_neurons = 100
    epochs = 20
    batch_size = 32
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'

    train, test, X_train, X_test, y_train, y_test = prepare_data(
        hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    targets = test[target_col][window_len:]
    preds = model.predict(X_test).squeeze()

    preds = test[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    return preds.values.tolist()
import flask
app = Flask(__name__)
@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        if flask.request.method == 'GET':
            coin = flask.request.args.get('coin')
            limit = flask.request.args.get('limit')
            data={}
            data['status']="true"
            data['message']="Data Fetched"
            data['data']=predict_data(coin,limit)
            
        return str(data)
    except:
        data={}
        data['status']="false"
        data['message']="Error"
        data['data']="null"
        return str(data)


if __name__ == "__main__":
   app.run(host='0.0.0.0')