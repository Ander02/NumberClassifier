import numpy as np
import tensorflow.keras as keras
import pandas as pd


def initialize_model():
    model = keras.Sequential()
    for i in range(7):
        model.add(keras.layers.Dense(units=128, activation="relu"))
    model.add(keras.layers.Dense(units=128, activation="relu"))
    model.add(keras.layers.Dense(units=10, activation="sigmoid"))
    return model


def train(model, database_path, save_path, batch_size=5000, epochs=400, optimizer="rmsprop", metrics=["accuracy", "Precision", "MeanSquaredError", "CategoricalAccuracy"]):

    data_frame = pd.read_csv(database_path)

    labels = data_frame["label"].to_numpy()

    fit_output = np.zeros([len(labels), 10])

    for i, label in enumerate(labels):
        fit_output[i][label] = 1

    values = data_frame.drop(labels="label", axis=1).to_numpy()

    #print("Training Values {}".format(data_frame["label"][:35000].value_counts()))
    #print("Validation Values {}".format(data_frame["label"][-7000:].value_counts()))

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    model.fit(x=values[:35000], y=fit_output[:35000],
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              validation_data=[values[-7000:], fit_output[-7000:]],
              verbose=1)

    model.save(save_path)
    print(model.summary())
