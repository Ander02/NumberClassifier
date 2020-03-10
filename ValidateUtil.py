import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def validate(model):
    dataset_dataframe = pd.read_csv("./database/train.csv")

    confusion_dataframe = pd.DataFrame(columns=["Label", "PredictedLabel", "PredictedAccuracy"])

    predicts = model.predict(dataset_dataframe.drop(labels="label", axis=1).to_numpy())

    labels = dataset_dataframe["label"]
    predicts_series = pd.Series(predicts[:, 0], name='PredictedLabel')
    predicts_acc = pd.Series(predicts[:, 1], name="PredictedAccuracy")
    df = pd.DataFrame([labels, predicts_series, predicts_acc])

    # df[df["PredictedLabel"] ]

    conf_matrix = tf.math.confusion_matrix(labels=labels.to_numpy(),
                                           predictions=(predicts.argmax(-1))).numpy()

    new_matriz = np.zeros([10, 10])
    for i, line in enumerate(conf_matrix):
        for j, column in enumerate(conf_matrix[i]):
            new_matriz[i][j] = column/len(labels[labels == j].to_numpy())

    print(pd.DataFrame(np.around(new_matriz * 100, decimals=2)))

    # data = np.zeros([10, 10])
    # for i in range(10):
    #     data[i] = np.zeros([10])
    #     for j in range(10):
    #         data[i][j] = np.count_nonzero(predicts[predicts[i, j] == np.argmax(predicts[i])])
    #
    # for i in range(10):
    #     print(i)
    #     for j in range(10):
    #         print(data[i][j])
    #
    # for i, label in enumerate(dataset_dataframe["label"].to_numpy()):
    #     predicted_label = np.argmax(predicts[i])
    #     new_row = pd.DataFrame({
    #         "Label": label,
    #         "PredictedLabel0": 0 == predicted_label,
    #         "PredictedLabel1": 1 == predicted_label,
    #         "PredictedAccuracy": predicts[i][predicted_label]
    #     }, index=[i])
    #     confusion_dataframe = confusion_dataframe.append(new_row)







