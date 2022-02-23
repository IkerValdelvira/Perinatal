import argparse
import os
import warnings
from itertools import cycle
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from numpy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tabulate import tabulate
from tensorflow import keras
from keras.models import load_model

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def make_compatible(new_items, standard_scaler):
    # Eliminar todas las clases
    if 'pesorec' in new_items.columns:
        new_items = new_items.drop('pesorec', 1)
    if 'lbw' in new_items.columns:
        new_items = new_items.drop('lbw', 1)
    if 'nbw' in new_items.columns:
        new_items = new_items.drop('nbw', 1)
    if 'hbw' in new_items.columns:
        new_items = new_items.drop('hbw', 1)
    if 'peson' in new_items.columns:
        new_items = new_items.drop('peson', 1)
    if 'premature' in new_items.columns:
        new_items = new_items.drop('premature', 1)

    # Estandarización mediante StandardScaler
    dataset_columns = new_items.columns
    dataset_index = new_items.index.values
    new_items_array = standard_scaler.transform(new_items)
    new_items = pd.DataFrame(new_items_array, index=dataset_index, columns=dataset_columns)

    return new_items


def make_predictions(new_items, model, algorithm, output_dir):
    file = open(os.path.join(output_dir, 'PREDICTIONS.txt'), "w")
    file.write("\nPredictions on new items:\n")

    if (algorithm == 'DNN'):
        pred = model.predict(new_items)
        pred = pred.argmax(axis=-1)
        i = 0
        results = []
        for index in new_items.index:
            pred_class = "LOW weight"
            if (pred[i] == 1.0):
                pred_class = "NORMAL weight"
            elif (pred[i] == 2.0):
                pred_class = "HIGH weight"
            results.append(["Item: " + str(index),
                            "Predicted: " + str(pred[i]) + " (" + pred_class + ")"])
            i += 1

        file.write(tabulate(results))
        print('Predictions on new items saved in: ' + str(os.path.join(output_dir, 'PREDICTIONS.txt')))

    if (algorithm == 'RF'):
        pred = model.predict(new_items)
        pred_prob = model.predict_proba(new_items)
        i = 0
        results = []
        for index in new_items.index:
            pred_class = "LOW weight"
            if (pred[i] == 1.0):
                pred_class = "NORMAL weight"
                prob = str(round((pred_prob[i][1] * 100), 2)) + "%"
            elif (pred[i] == 2.0):
                pred_class = "HIGH weight"
                prob = str(round((pred_prob[i][2] * 100), 2)) + "%"
            else:
                prob = str(round((pred_prob[i][0] * 100), 2)) + "%"
            results.append(["Item: " + str(index),
                            "Predicted: " + str(pred[i]) + "(" + pred_class + ")",
                            "Probability: " + prob])
            i += 1

        file.write(tabulate(results))
        print('Predictions on new items saved in: ' + str(os.path.join(output_dir, 'PREDICTIONS.txt')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to make predictions on new input items using 'pesorec' predictive model. Usage example: $python make_predictions_pesorec.py new_items.csv pathTo/ModelFolder -o pathTo/Predictions")
    parser.add_argument("new_items", help="CSV file which contains new items to predict. For example: new_items.csv")
    parser.add_argument("model_folder",
                        help="Path to folder where a 'pesorec' predictive model and all necessary files are saved. For example: pathTo/ModelFolder")
    parser.add_argument("-o", "--output_dir",
                        help="Path to the output directory for creating the file with new items predictions.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    new_items_path = args['new_items']
    model_folder = args['model_folder']
    output_dir = args['output_dir']

    # Read dataset, model and necessary files
    print('\nReading CSV file which contains new items to predict from: ' + str(new_items_path))
    new_items = pd.read_csv(new_items_path, index_col=0)

    algorithm = ''
    for filename in os.listdir(model_folder):
        f = os.path.join(model_folder, filename)
        name, extension = os.path.splitext(f)
        if (extension == '.joblib'):
            print('\nReading RandomForest predictive model from: ' + str(f))
            model = joblib.load(f)
            algorithm = 'RF'
        elif (extension == '.h5'):
            print('\nReading DeepNeuralNetwork predictive model from: ' + str(f))
            model = load_model(f)
            algorithm = 'DNN'
        elif (extension == '.bin'):
            print('\nReading data standardization model from: ' + str(f))
            standard_scaler = joblib.load(f)

    print('\nMaking predictions on new input items...')
    new_items_compatible = make_compatible(new_items, standard_scaler)
    make_predictions(new_items_compatible, model, algorithm, output_dir)


