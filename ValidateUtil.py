import pandas as pd
import numpy as np
import tensorflow.keras as keras

def validate(model):
    dataset_dataframe = pd.read_csv("./database/train.csv")[:10]

    confusion_dataframe = pd.DataFrame(columns=["Label", "PredictedLabel", "PredictedAccuracy"])

    predicts = model.predict(dataset_dataframe.drop(labels="label", axis=1).to_numpy())

    for i, label in enumerate(dataset_dataframe["label"].to_numpy()):
        predicted_label = np.argmax(predicts[i])
        new_row = pd.DataFrame({
            "Label": label,
            "PredictedLabel": predicted_label,
            "PredictedAccuracy": predicts[i][predicted_label]
        })
        confusion_dataframe += new_row

    print(confusion_dataframe)





