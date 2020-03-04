import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def test(model):

    test_database_path = "./database/test.csv"
    data_frame = pd.read_csv(test_database_path).to_numpy()
    np.random.shuffle(data_frame)
    predicts = model.predict(data_frame)

    for i, line in enumerate(data_frame):
        img = np.reshape(line, [28, 28])
        plt.imshow(img)

        title_text = ""
        for j, predict in enumerate(predicts[i]):
            title_text += "{}: {:.3f}% | ".format(j, predict * 100)
        plt.title(title_text, fontdict={"fontsize": 8})
        plt.show()

    plt.close()
