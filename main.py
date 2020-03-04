import pandas as pd
import os
from ModelUtil import initialize_model, train
import tensorflow.keras as keras
from TestUtil import test
from ValidateUtil import validate

if __name__ == "__main__":
    
    database_path = "./database/train.csv"
    save_path = "./numbers.h5"

    if not os.path.exists(save_path):
        model = initialize_model()
        train(model=model, database_path=database_path, save_path=save_path)

    model = keras.models.load_model(save_path)
    #test(model)
    validate(model)