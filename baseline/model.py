from keras.layers import LSTM, Dense
from keras.models import Sequential


def lstm_model(input_shape=(40, 2048), recurrent_units=32, num_classes=9, stack=0):
    model = Sequential()
    for _ in range(stack):
        model.add(LSTM(recurrent_units, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(recurrent_units, input_shape=input_shape))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='RMSProp',
        metrics=['accuracy']
    )

    return model
