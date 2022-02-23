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
from matplotlib.ticker import PercentFormatter
from numpy import interp
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tabulate import tabulate
from tensorflow import keras

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def split_train_dev_test(dataset, output_dir):
    # Solo se va a usar la clase 'peson' --> Borrar 'lbw', 'nbw', 'hbw' y 'premature'
    dataset = dataset.drop(['lbw', 'nbw', 'hbw', 'premature'], 1)

    # Train(70%) / Dev(15%) / Test(15%)
    labels = dataset['peson']
    stratify_labels = dataset['pesorec']
    data = dataset.drop('peson', axis=1)
    X_train_dev, X_test, y_train_dev, y_test, stratify_labels_train_dev, stratify_labels_test = train_test_split(data,
                                                                                                                 labels,
                                                                                                                 stratify_labels,
                                                                                                                 test_size=0.15,
                                                                                                                 random_state=42,
                                                                                                                 shuffle=True,
                                                                                                                 stratify=stratify_labels)
    X_train, X_dev, y_train, y_dev, stratify_labels_train, stratify_labels_dev = train_test_split(X_train_dev,
                                                                                                  y_train_dev,
                                                                                                  stratify_labels_train_dev,
                                                                                                  test_size=(
                                                                                                              len(y_test) / len(
                                                                                                          y_train_dev)),
                                                                                                  random_state=42,
                                                                                                  shuffle=True,
                                                                                                  stratify=stratify_labels_train_dev)

    X_train['peson'] = y_train
    X_dev['peson'] = y_dev
    X_test['peson'] = y_test

    train = X_train.drop('pesorec', 1)
    dev = X_dev.drop('pesorec', 1)
    test = X_test.drop('pesorec', 1)

    train.to_csv(os.path.join(output_dir, "train.csv"))
    print("\nTrain set saved in : " + os.path.join(output_dir, "train.csv"))
    dev.to_csv(os.path.join(output_dir, "dev.csv"))
    print("Dev set saved in : " + os.path.join(output_dir, "dev.csv"))
    test.to_csv(os.path.join(output_dir, "test.csv"))
    print("Test set saved in : " + os.path.join(output_dir, "test.csv"))

    return train, dev, test


def experiment_rf(train, dev, output_dir):
    file = open(os.path.join(output_dir, "REPORT_RandomForest_peson.txt"), "w")
    file.write("RandomForest model: 'peson' prediction:\n\n")

    # Features/Labels
    y_train = train['peson']
    X_train = train.drop('peson', axis=1)
    feature_names = X_train.columns
    y_dev = dev['peson']
    X_dev = dev.drop('peson', axis=1)
    indices_dev = dev.index.tolist()

    # Predictions
    print("\n[RF] Training RandomForest model...")
    model = RandomForestRegressor(max_depth=25)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, os.path.join(output_dir, "RF_MODEL_peson.joblib"), compress=3)
    print('[RF] Trained model saved in: ' + os.path.join(output_dir, "RF_MODEL_peson.joblib"))

    print('\n[RF] Evaluating model...')

    # Classification report (Dev)
    y_pred_dev = model.predict(X_dev)
    file.write("\nDev dataset evaluation:")
    file.write('\n\tMean Absolute Error (MAE): ' + str(round(metrics.mean_absolute_error(y_dev, y_pred_dev),4)))
    file.write('\n\tMean Squared Error (MSE): ' + str(round(metrics.mean_squared_error(y_dev, y_pred_dev),4)))
    file.write('\n\tRoot Mean Squared Error (RMSE): ' + str(round(metrics.mean_squared_error(y_dev, y_pred_dev, squared=False),4)))
    file.write('\n\tMean Absolute Percentage Error (MAPE): ' + str(round(metrics.mean_absolute_percentage_error(y_dev, y_pred_dev),4)))

    print('[RF] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_RandomForest_peson.txt"))

    # Predictions plot (Dev)
    plt.scatter(y_dev, y_pred_dev, s=2)
    plt.xlabel('True Values (grams)')
    plt.ylabel('Predictions (grams)')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-7000, 7000], [-7000, 7000], color='black', linestyle='dashed')
    plt.title("Dev --> 'peson' predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Predictions(Dev).png"))
    print('[RF] Dev preditions plot saved in: ' + os.path.join(output_dir, "Predictions(Dev).png"))
    plt.close()

    error = y_pred_dev - y_dev
    plt.hist(error, bins=100, weights=np.ones(len(error)) / len(error))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel("Prediction Error")
    plt.ylabel("Frecuency (%)")
    plt.title("Dev --> 'peson' predictions error frecuency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PredictionsError(Dev).png"))
    print('[RF] Dev preditions error frenquency plot saved in: ' + os.path.join(output_dir, "Predictions(Dev).png"))
    plt.close()

    # Feature importance (Mean decrease in impurity)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "FeatureImportanceMDI_peson.png"))
    print('[RF] Feature importance based on mean decrease in impurity saved in: ' + os.path.join(output_dir, "FeatureImportanceMDI_peson.png"))
    plt.close()


def experiment_dnn(train, dev, output_dir):
    # Features/Labels
    y_train_val = train['peson']
    X_train_val = train.drop('peson', axis=1)
    y_dev = dev['peson']
    X_dev = dev.drop('peson', axis=1)
    indices_dev = dev.index.tolist()
    n_features = len(X_train_val.columns)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.2, random_state=42,
                                                      shuffle=True)

    # Standardization
    print("\n[DNN] Standardization of datasets...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_dev = scaler.transform(X_dev)
    joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_peson.bin'), compress=True)
    print("[DNN] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_peson.bin')))

    print("\n[DNN] Training DNN model...")
    input_shape = (n_features,)

    batch_size = 64
    epochs = 1000

    # (input*2/3)+output, (input+output)/2
    """model = keras.models.Sequential()
    model.add(keras.layers.Dense(int((n_features * 2 / 3) + 1), input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(int((n_features + 1) / 2), activation='relu'))
    model.add(keras.layers.Dense(1))"""

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                                                            tf.keras.metrics.MeanSquaredError(name="mse"),
                                                            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                                                            tf.keras.metrics.MeanAbsolutePercentageError(name="mape")])

    file = open(os.path.join(output_dir, "REPORT_DNN_peson.txt"), "w")
    file.write("DNN model: 'peson' prediction\n\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))

    csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    train = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                      validation_data=(X_val, y_val), callbacks=[csv_logger, early_stop])

    print('[DNN] Trained model saved in: ' + os.path.join(output_dir, "DNN_MODEL_peson.h5"))
    model.save(os.path.join(output_dir, "DNN_MODEL_peson.h5"))

    # Train and validation MAE and MSE plots
    hist = pd.DataFrame(train.history)
    hist['epoch'] = train.epoch

    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'], label='Training Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Validation Error')
    plt.legend()
    plt.title('Training and validation MAE error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Training_MAEerror.png"))
    plt.close()
    print('[DNN] Model MAE error history of training saved in: ' + os.path.join(output_dir, "Training_MAEerror.png"))

    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'], label='Training Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Validation Error')
    plt.legend()
    plt.title('Training and validation MSE error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Training_MSEerror.png"))
    plt.close()
    print('[DNN] Model MSE error history of training saved in: ' + os.path.join(output_dir, "Training_MSEerror.png"))

    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Square Error')
    plt.plot(hist['epoch'], hist['rmse'], label='Training Error')
    plt.plot(hist['epoch'], hist['val_rmse'], label='Validation Error')
    plt.legend()
    plt.title('Training and validation RMSE error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Training_RMSEerror.png"))
    plt.close()
    print('[DNN] Model RMSE error history of training saved in: ' + os.path.join(output_dir, "Training_RMSEerror.png"))

    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Percentaje Error')
    plt.plot(hist['epoch'], hist['mape'], label='Training Error')
    plt.plot(hist['epoch'], hist['val_mape'], label='Validation Error')
    plt.legend()
    plt.title('Training and validation MAPE error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Training_MAPEerror.png"))
    plt.close()
    print('[DNN] Model MSE error history of training saved in: ' + os.path.join(output_dir, "Training_MAPEerror.png"))

    # DEV evaluation
    print('\n[DNN] Evaluating model on Dev set...')
    dev_eval = model.evaluate(X_dev, y_dev, verbose=1)
    file.write("\n\nDev dataset evaluation:\n")
    file.write('\tMean Absolute Error (MAE): ' + str(round(dev_eval[1], 4)) + "\n")
    file.write('\tMean Squared Error (MSE): ' + str(round(dev_eval[2], 4)) + "\n")
    file.write('\tRoot Mean Squared Error (RMSE): ' + str(round(dev_eval[3], 4)) + "\n")
    file.write('\tMean Absolute Percentage Error (MAPE): ' + str(round(dev_eval[4], 4)) + "\n")

    print('[DNN] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_DNN_peson.txt"))

    y_pred_dev = model.predict(X_dev).flatten()

    # Predictions plot (Dev)
    plt.scatter(y_dev, y_pred_dev, s=2)
    plt.xlabel('True Values (grams)')
    plt.ylabel('Predictions (grams)')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-7000, 7000], [-7000, 7000], color='black', linestyle='dashed')
    plt.title("Dev --> 'peson' predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Predictions(Dev).png"))
    print('[DNN] Dev preditions plot saved in: ' + os.path.join(output_dir, "Predictions(Dev).png"))
    plt.close()

    error = y_pred_dev - y_dev
    plt.hist(error, bins=100, weights=np.ones(len(error)) / len(error))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel("Prediction Error")
    plt.ylabel("Frecuency (%)")
    plt.title("Dev --> 'peson' predictions error frecuency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PredictionsError(Dev).png"))
    print('[DNN] Dev preditions error frenquency plot saved in: ' + os.path.join(output_dir, "Predictions(Dev).png"))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to perform 'peson' regression using RandomForestRegressor and DNN on Perinatal dataset. Data is splitted in Train(70%), Dev(15%), Test(15%). Usage example: $python experiments_peson.py dataPerinatal_predictions.csv -o pathTo/ExperimentsPeson")
    parser.add_argument("input_perinatal_dataset",
                        help="Path to file with input Perinatal dataset. For example: 'dataPerinatal_predictions.csv'.")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created Train/Dev/Test sets and results of regression models. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_perinatal_dataset = args['input_perinatal_dataset']
    output_dir = args['output_dir']

    print("\nReading Perinatal dataset for 'peson' regression from: " + str(input_perinatal_dataset))
    dataset_pesorec = pd.read_csv(input_perinatal_dataset, index_col=0)

    print("\nPerforming Train(70%) / Dev(15%) / Test(15%) split of Perinatal dataset...")
    Path(os.path.join(output_dir, "Train_Dev_Test")).mkdir(parents=True, exist_ok=True)
    train, dev, test = split_train_dev_test(dataset_pesorec, os.path.join(output_dir, "Train_Dev_Test"))

    Path(os.path.join(output_dir, "Results")).mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(output_dir, "Results")

    print("\nTraining 'peson' regression models...")
    Path(os.path.join(output_dir, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "RandomForest")
    experiment_rf(train, dev, output_dir_aux)
    Path(os.path.join(output_dir, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "DeepNeuralNetwork")
    experiment_dnn(train, dev, output_dir_aux)

