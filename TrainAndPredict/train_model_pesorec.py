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

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess(train, resample, algorithm, output_dir):
    # Eliminar todas las clases menos 'pesorec'
    if 'lbw' in train.columns:
        train = train.drop('lbw', 1)
    if 'nbw' in train.columns:
        train = train.drop('nbw', 1)
    if 'hbw' in train.columns:
        train = train.drop('hbw', 1)
    if 'peson' in train.columns:
        train = train.drop('peson', 1)
    if 'premature' in train.columns:
        train = train.drop('premature', 1)

    # Features/Labels
    y_train = train['pesorec']
    X_train = train.drop('pesorec', axis=1)
    column_names = X_train.columns
    indices = train.index.tolist()
    total_items = len(y_train)

    # Standardization
    print("\n\tStandardization of dataset...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=column_names, index=indices)
    if (algorithm == 'RF'):
        joblib.dump(scaler, os.path.join(output_dir, "RandomForest", 'std_scaler_pesorec.bin'), compress=True)
        print("\tStandard scaler saved in: " + str(os.path.join(output_dir, "RandomForest", 'std_scaler_pesorec.bin')))
    elif (algorithm == 'DNN'):
        joblib.dump(scaler, os.path.join(output_dir, "DeepNeuralNetwork", 'std_scaler_pesorec.bin'), compress=True)
        print("\tStandard scaler saved in: " + str(os.path.join(output_dir, "DeepNeuralNetwork", 'std_scaler_pesorec.bin')))
    else:
        joblib.dump(scaler, os.path.join(output_dir, "RandomForest", 'std_scaler_pesorec.bin'), compress=True)
        print("\tStandard scaler saved in: " + str(os.path.join(output_dir, "RandomForest", 'std_scaler_pesorec.bin')))
        joblib.dump(scaler, os.path.join(output_dir, "DeepNeuralNetwork", 'std_scaler_pesorec.bin'), compress=True)
        print("\tStandard scaler saved in: " + str(os.path.join(output_dir, "DeepNeuralNetwork", 'std_scaler_pesorec.bin')))

    # Resampling TRAIN set
    if (resample != ''):
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        print("\n\tDistribution before resampling:")
        print("\t\tCount of 'LOW' weight items: " + str(occurrences[0]) + " (" + str(round(occurrences[0] / total_items * 100)) + "%)")
        print("\t\tCount of 'NORMAL' weight items: " + str(occurrences[1]) + " (" + str(round(occurrences[1] / total_items * 100)) + "%)")
        print("\t\tCount of 'LOW' weight items: " + str(occurrences[2]) + " (" + str(round(occurrences[2] / total_items * 100)) + "%)")

    if (resample == 'OS'):  # Oversampling
        print("\n\tOversampling not majority classes ('LOW' and 'HIGH') of training set...")
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    if (resample == 'SMOTE'):  # Oversampling with SMOTE
        print("\n\tOversampling minority class of training set with SMOTE...")
        oversampler = SMOTE(sampling_strategy='minority')
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    elif (resample == 'US'):  # Undersampling
        print("\n\tUndersampling not minority classes ('NORMAL' and 'HIGH') of training set...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    elif (resample == 'TomekLinks'):  # Undersampling with TomekLinks
        print("\n\tUndersampling mayority class of training set with TomekLinks...")
        undersampler = TomekLinks(sampling_strategy='majority')
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        indices_selected = undersampler.sample_indices_
        print(indices_selected)

    elif (resample == 'OUS'):  # Oversampling(0.1)/Undersampling(0.5)
        print("\n\tOversampling not majority classes ('LOW' and 'HIGH') of training set by 10%...")
        oversampler = RandomOverSampler(random_state=42,
                                        sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]),
                                                           2: int(occurrences[2] * 1.1)})
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        print("\tUndersampling majority class ('NORMAL') of training set by 50%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        undersampler = RandomUnderSampler(random_state=42,
                                          sampling_strategy={0: int(occurrences[0]), 1: int(occurrences[1] * 0.5),
                                                             2: int(occurrences[2])})
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    if (resample != ''):
        total_items = len(y_train)
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        print("\n\tDistribution after resampling:")
        print("\t\tCount of 'LOW' weight items: " + str(occurrences[0]) + " (" + str(round(occurrences[0] / total_items * 100)) + "%)")
        print("\t\tCount of 'NORMAL' weight items: " + str(occurrences[1]) + " (" + str(round(occurrences[1] / total_items * 100)) + "%)")
        print("\t\tCount of 'LOW' weight items: " + str(occurrences[2]) + " (" + str(round(occurrences[2] / total_items * 100)) + "%)")

    X_train['pesorec'] = y_train
    return X_train


def train_rf(train, output_dir):
    # Features/Labels
    y_train = train['pesorec']
    X_train = train.drop('pesorec', axis=1)

    # Predictions
    print("\n\t[RF] Training RandomForest model...")
    classifier = RandomForestClassifier(max_depth=25)
    classifier.fit(X_train, y_train)

    # Save the model
    joblib.dump(classifier, os.path.join(output_dir, "RF_MODEL_pesorec.joblib"), compress=3)
    print('\t[RF] Trained model saved in: ' + os.path.join(output_dir, "RF_MODEL_pesorec.joblib"))


def train_dnn(train, output_dir):
    # Features/Labels
    y_train = train['pesorec']
    X_train = train.drop('pesorec', axis=1)
    n_features = len(X_train.columns)

    print("\n\t[DNN] Training DNN model...")
    input_shape = (n_features,)
    n_classes = 3

    y_train = np_utils.to_categorical(y_train)

    batch_size = 64
    epochs = 50

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(160, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(320, activation='sigmoid'))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.0001),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.AUC(name="AUC")])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

    train = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                      validation_split=0.2, callbacks=[reduce_lr, early_stop])

    print('\t[DNN] Trained model saved in: ' + os.path.join(output_dir, "DNN_MODEL_pesorec.h5"))
    model.save(os.path.join(output_dir, "DNN_MODEL_pesorec.h5"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Script to train a 'pesorec' classification model with the entered dataset, and using RandomForest and/or optimized DNN. Resampling techniques can be applied like oversampling, undersampling and over/undersampling. Usage example: $python train_model_pesorec.py dataPerinatal.csv -a DNN -o pathTo/ModelFolder")
    parser.add_argument("input_pesorec_train",
                        help="Path to file with input training dataset for 'pesorec' classification. For example: 'dataPerinatal.csv'.")
    parser.add_argument("-rs", "--resample_method",
                        help="Method to resample data of training set: oversampling not majority classes ('LOW' and 'HIGH') [OS], undersampling not minority classes ('LOW' and 'NORMAL') [US] or oversampling not majority classes ('LOW' and 'HIGH') by a 10 percent and undersampling majority class ('NORMAL') by a 50 percent [OUS], oversampling minority class using SMOTE [SMOTE] or undersampling majority class using TomekLinks [TomekLinks]. Default option: empty.",
                        default='')
    parser.add_argument("-a", "--algorithm",
                        help="Algortithm for make predictions: RandomForest [RF], DeepNeuralNetwork [DNN] or both [both]. Default option: [DNN].",
                        default='DNN')
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created classification models.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_pesorec_train = args['input_pesorec_train']
    resample = args['resample_method']
    algorithm = args['algorithm']
    output_dir = args['output_dir']

    if resample not in ['OS', 'US', 'OUS', 'SMOTE', 'TomekLinks', '']:
        parser.error("'resample_method' value must be [OS], [US], [OUS], [SMOTE], [TomekLinks] or empty.")

    if algorithm not in ['RF', 'DNN', 'both']:
        parser.error("'algorithm' value must be [RF], [CNN] or [both].")

    print("\nReading Train dataset for 'pesorec' classification from: " + str(input_pesorec_train))
    train = pd.read_csv(input_pesorec_train, index_col=0)

    Path(os.path.join(output_dir, "Models")).mkdir(parents=True, exist_ok=True)
    if (algorithm == 'RF'):
        Path(os.path.join(output_dir, "Models", "RandomForest")).mkdir(parents=True, exist_ok=True)
    elif (algorithm == 'DNN'):
        Path(os.path.join(output_dir, "Models", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    else:
        Path(os.path.join(output_dir, "Models", "RandomForest")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_dir, "Models", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)

    print("\nPreprocessing dataset before training...")
    train = preprocess(train, resample, algorithm, os.path.join(output_dir, "Models"))

    print("\nTraining 'pesorec' classification models...")
    if (algorithm == 'RF'):
        train_rf(train, os.path.join(output_dir, "Models", "RandomForest"))
    elif (algorithm == 'DNN'):
        train_dnn(train, os.path.join(output_dir, "Models", "DeepNeuralNetwork"))
    else:
        train_rf(train, os.path.join(output_dir, "Models", "RandomForest"))
        train_dnn(train, os.path.join(output_dir, "Models", "DeepNeuralNetwork"))

