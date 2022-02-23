import argparse
import json
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

def split_train_dev_test(dataset, output_dir):
    # Eliminación de algunas features
    dataset = dataset.drop(['IDENTHOGAR', 'PROXY_0', 'PROXY_1', 'PROXY_2', 'PROXY_2b', 'PROXY_3b', 'PROXY_4', 'PROXY_5', 'E2_1c', 'E2_1d', 'F7', 'G25b_2', 'G25c_2', 'G25b_3', 'G25c_3', 'G25b_11', 'G25c_11', 'G25b_13', 'G25c_13', 'G25b_19', 'G25c_19', 'G25b_22', 'G25c_22', 'G25b_23', 'G25c_23', 'G25a_30', 'G25b_30', 'G25c_30', 'O73', 'O84_8', 'O84_9', 'P87_3b', 'P87_5b', 'P87_8b', 'P87_9b', 'P87_10b', 'P87_15b', 'P87_16b', 'P87_17b', 'P87_21b', 'P87_22b', 'U120FZ', 'U120CANTFZ', 'W128Cer', 'W128Cer_1', 'W128Cer_2', 'W128Cer_3', 'W128Cer_4', 'W128Cer_5', 'W128Cer_6', 'W128Cer_7', 'W128Vin', 'W128Vin_1', 'W128Vin_2', 'W128Vin_3', 'W128Vin_4', 'W128Vin_5', 'W128Vin_6', 'W128Vin_7', 'W128Vermut', 'W128Vermut_1', 'W128Vermut_2', 'W128Vermut_3', 'W128Vermut_4', 'W128Vermut_5', 'W128Vermut_6', 'W128Vermut_7', 'W128Lic', 'W128Lic_1', 'W128Lic_2', 'W128Lic_3', 'W128Lic_4', 'W128Lic_5', 'W128Lic_6', 'W128Lic_7', 'W128Comb', 'W128Comb_1', 'W128Comb_2', 'W128Comb_3', 'W128Comb_4', 'W128Comb_5', 'W128Comb_6', 'W128Comb_7', 'W128Sidra', 'W128Sidra_1', 'W128Sidra_2', 'W128Sidra_3', 'W128Sidra_4', 'W128Sidra_5', 'W128Sidra_6', 'W128Sidra_7', 'W129', 'FACTORADULTO', 'CMD1', 'CMD2', 'CMD3'], 1)

    # Train(70%) / Dev(15%) / Test(15%)
    labels = dataset['alcohol']
    data = dataset.drop('alcohol', axis=1)
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(data, labels,
                                                                test_size=0.15, random_state=42,
                                                                shuffle=True, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev,
                                                      test_size=(len(y_test) / len(y_train_dev)),
                                                      random_state=42, shuffle=True,
                                                      stratify=y_train_dev)

    X_train['alcohol'] = y_train
    X_dev['alcohol'] = y_dev
    X_test['alcohol'] = y_test

    train = X_train
    dev = X_dev
    test = X_test

    train.to_csv(os.path.join(output_dir, "train.csv"))
    print("Train set saved in : " + os.path.join(output_dir, "train.csv"))
    dev.to_csv(os.path.join(output_dir, "dev.csv"))
    print("Dev set saved in : " + os.path.join(output_dir, "dev.csv"))
    test.to_csv(os.path.join(output_dir, "test.csv"))
    print("Test set saved in : " + os.path.join(output_dir, "test.csv"))

    return train, dev, test


def experiment_rf(train, dev, output_dir):
    file = open(os.path.join(output_dir, "REPORT_RandomForest_alcohol.txt"), "w")
    file.write("RandomForest model: 'alcohol' prediction:\n\n")

    # Features/Labels
    y_train = train['alcohol']
    X_train = train.drop('alcohol', axis=1)
    feature_names = X_train.columns
    y_dev = dev['alcohol']
    X_dev = dev.drop('alcohol', axis=1)
    indices_dev = dev.index.tolist()

    # Imputing missing values
    print("\n[RF] Imputing missing values with mode of features (categorical)...")
    imputation_dict = X_train.mode().iloc[0].to_dict()
    X_train = X_train.fillna(X_train.mode().iloc[0])
    for column in imputation_dict.keys():
        X_dev[column] = X_dev[column].fillna(imputation_dict[column])
    imputation_dict_file = open(os.path.join(output_dir, "imputation_dict_alcohol.json"), "w")
    json.dump(imputation_dict, imputation_dict_file)
    imputation_dict_file.close()
    print("[RF] Missing value imputer saved in: " + str(os.path.join(output_dir, "imputation_dict_alcohol.json")))

    # Standardization
    print("\n[RF] Standardization of datasets...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)
    joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_alcohol.bin'), compress=True)
    print("[RF] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_alcohol.bin')))

    """# Select 25 features with highest Pearson Correlation
    print("\n[RF] Selecting only 25 features with highest Pearson Correlation...")
    train_cor = pd.DataFrame(X_train, columns=feature_names)
    train_cor['alcohol'] = y_train
    cor = train_cor.corr()
    cor_target_25 = abs(cor["alcohol"]).sort_values(ascending=False).head(26)
    cor_features_25 = cor_target_25.index.tolist()
    X_train = train_cor[cor_features_25].drop('alcohol', 1)
    cor_features_25.remove('alcohol')
    X_dev = pd.DataFrame(X_dev, columns=feature_names)
    X_dev = X_dev[cor_features_25]
    print("[RF] 25 features with highest Pearson Correlation: " + str(cor_features_25))"""

    # Undersampling TRAIN set
    print("\n[RF] Undersampling mayority class of training set by a 50%...")
    undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)

    # Predictions
    print("\n[RF] Training RandomForest model...")
    classifier = RandomForestClassifier(max_depth=25)
    classifier.fit(X_train, y_train)
    y_pred_dev = classifier.predict(X_dev)

    # Save the model
    joblib.dump(classifier, os.path.join(output_dir, "RF_MODEL_alcohol.joblib"), compress=3)
    print('[RF] Trained model saved in: ' + os.path.join(output_dir, "RF_MODEL_alcohol.joblib"))

    print('\n[RF] Evaluating model...')

    # Classification report (Dev)
    file.write("\nClassification Report (Dev):\n" + classification_report(y_dev, y_pred_dev))

    # Confusion matrix (Dev)
    conf_mat_dev = confusion_matrix(y_dev, y_pred_dev)
    tn, fp, fn, tp = conf_mat_dev.ravel()
    specificity_test = tn / (tn + fp)
    sensitivity_test = tp / (tp + fn)
    file.write("\nspecificity\t\t" + str(round(specificity_test, 2)))
    file.write("\nsensitivity\t\t" + str(round(sensitivity_test, 2)) + "\n")
    df_cm_dev = pd.DataFrame(conf_mat_dev, index=['No', 'Yes'], columns=['No', 'Yes'])
    sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix (Dev), 'alcohol' prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfusionMatrix_alcohol.png"))
    plt.close()
    print('[RF] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_alcohol.png"))

    # Dev predictions probabilities
    i = 0
    results = []
    for y in y_dev:
        success = "YES"
        if (y != y_pred_dev[i]):
            success = "NO"
        results.append(["Instance: " + str(indices_dev[i]), "Class: " + str(y),
                        "Predicted: " + str(y_pred_dev[i]),
                        "Success: " + success])
        i += 1
    file.write("\n\nPredictions on Dev set:\n")
    file.write(tabulate(results))

    # ROC curve and AUC (Dev)
    # roc curve for models
    rf_data = {}
    predictions_dev = classifier.predict_proba(X_dev)
    fpr, tpr, thresholds = roc_curve(y_dev, predictions_dev[:, 1], pos_label=1)
    auc_metric = auc(fpr, tpr)
    rf_data['fpr_dev'] = fpr
    rf_data['tpr_dev'] = tpr
    rf_data['auc_dev'] = auc_metric
    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_dev))]
    p_fpr, p_tpr, _ = roc_curve(y_dev, random_probs, pos_label=1)
    # plot roc curves
    plt.plot(fpr, tpr, color='red',
             label='RandomForest (AUC = %0.4f)' % auc_metric)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve (Dev), 'alcohol' prediction")
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_alcohol.png"))
    print('[RF] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_alcohol.png"))
    plt.close()

    print('[RF] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_RandomForest_alcohol.txt"))

    # Feature importance (Mean decrease in impurity)
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    forest_importances_30 = forest_importances.head(30)
    fig, ax = plt.subplots()
    forest_importances_30.plot.bar(ax=ax)
    ax.set_title("Most important 30 features using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "FeatureImportanceMDI_alcohol.png"))
    print('[RF] Feature importance based on mean decrease in impurity saved in: ' + os.path.join(output_dir,
                                                                                                 "FeatureImportanceMDI_alcohol.png"))

    """# Feature importance (Mean decrease in impurity)
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=cor_features_25).sort_values(ascending=False)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "FeatureImportanceMDI_alcohol.png"))
    print('[RF] Feature importance based on mean decrease in impurity saved in: ' + os.path.join(output_dir,
                                                                                                 "FeatureImportanceMDI_alcohol.png"))"""
    plt.close()

    return rf_data


def experiment_dnn(train, dev, output_dir):
    # Features/Labels
    y_train_val = train['alcohol']
    X_train_val = train.drop('alcohol', axis=1)
    indices_train = train.index.tolist()
    column_names = X_train_val.columns
    y_dev = dev['alcohol']
    X_dev = dev.drop('alcohol', axis=1)
    indices_dev = dev.index.tolist()
    n_features = len(X_train_val.columns)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.2, random_state=42,
                                                      shuffle=True, stratify=y_train_val)

    # Imputing missing values
    print("\n[DNN] Imputing missing values with mode of features (categorical)...")
    imputation_dict = X_train.mode().iloc[0].to_dict()
    X_train = X_train.fillna(X_train.mode().iloc[0])
    for column in imputation_dict.keys():
        X_val[column] = X_val[column].fillna(imputation_dict[column])
        X_dev[column] = X_dev[column].fillna(imputation_dict[column])
    imputation_dict_file = open(os.path.join(output_dir, "imputation_dict_alcohol.json"), "w")
    json.dump(imputation_dict, imputation_dict_file)
    imputation_dict_file.close()
    print("[DNN] Missing value imputer saved in: " + str(os.path.join(output_dir, "imputation_dict_alcohol.json")))

    # Standardization
    print("\n[DNN] Standardization of datasets...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_dev = scaler.transform(X_dev)
    joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_alcohol.bin'), compress=True)
    print("[DNN] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_alcohol.bin')))

    """# Select 25 features with highest Pearson Correlation
    print("\n[DNN] Selecting only 25 features with highest Pearson Correlation...")
    train_cor = pd.DataFrame(X_train, columns=column_names)
    train_cor['alcohol'] = y_train
    cor = train_cor.corr()
    cor_target_25 = abs(cor["alcohol"]).sort_values(ascending=False).head(26)
    cor_features_25 = cor_target_25.index.tolist()
    X_train = train_cor[cor_features_25].drop('alcohol', 1)
    cor_features_25.remove('alcohol')
    X_val = pd.DataFrame(X_val, columns=column_names)
    X_val = X_val[cor_features_25]
    X_dev = pd.DataFrame(X_dev, columns=column_names)
    X_dev = X_dev[cor_features_25]
    print("[DNN] 25 features with highest Pearson Correlation: " + str(cor_features_25))"""

    # Undersampling TRAIN set
    print("\n[DNN] Undersampling mayority class of training set by a 50%...")
    undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)

    print("\n[DNN] Training DNN model...")
    input_shape = (n_features,)
    #input_shape = (len(cor_features_25),)

    batch_size = 64
    epochs = 50

    # (input*2/3)+output, (input+output)/2
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(int((n_features * 2 / 3) + 1), input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(int((n_features + 1) / 2), activation='relu'))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.AUC(name="AUC")])

    file = open(os.path.join(output_dir, "REPORT_DNN_alcohol.txt"), "w")
    file.write("DNN model: 'alcohol' prediction\n\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))

    csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

    train = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                      validation_data=(X_val, y_val), callbacks=[csv_logger, reduce_lr, early_stop])

    print('[DNN] Trained model saved in: ' + os.path.join(output_dir, "DNN_MODEL_alcohol.h5"))
    model.save(os.path.join(output_dir, "DNN_MODEL_alcohol.h5"))

    # DEV evaluation
    print('\n[DNN] Evaluating model on Dev set...')
    dev_eval = model.evaluate(X_dev, y_dev, verbose=1)
    file.write("\n\nDev dataset evaluation:\n")
    file.write('\tDev loss: ' + str(round(dev_eval[0], 4)) + "\n")
    file.write('\tDev accuracy: ' + str(round(dev_eval[1], 4)) + "\n")
    file.write('\tDev recall: ' + str(round(dev_eval[2], 4)) + "\n")
    file.write('\tDev precision: ' + str(round(dev_eval[3], 4)) + "\n")
    file.write('\tDev AUC: ' + str(round(dev_eval[4], 4)) + "\n\n")

    predicted_classes_dev = np.where(model.predict(X_dev) > 0.5, 1, 0)

    # Classification report (Dev)
    file.write("\nClassification Report (Dev):\n" + classification_report(y_dev, predicted_classes_dev))

    # Confusion matrix (Dev)
    conf_mat_dev = confusion_matrix(y_dev, predicted_classes_dev)
    tn, fp, fn, tp = conf_mat_dev.ravel()
    specificity_dev = tn / (tn + fp)
    sensitivity_dev = tp / (tp + fn)
    file.write("\nspecificity\t\t" + str(specificity_dev))
    file.write("\nsensitivity\t\t" + str(sensitivity_dev) + "\n\n")
    df_cm_dev = pd.DataFrame(conf_mat_dev, index=['No', 'Yes'], columns=['No', 'Yes'])
    sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix (Dev), 'alcohol' prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfusionMatrix_alcohol.png"))
    plt.close()
    print('[DNN] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_alcohol.png"))

    # Dev predictions probabilities
    i = 0
    results = []
    for y in y_dev:
        success = "YES"
        if (y != predicted_classes_dev[i][0]):
            success = "NO"

        results.append(["Instance: " + str(indices_dev[i]), "Class: " + str(int(y)),
                        "Predicted: " + str(predicted_classes_dev[i][0]),
                        "Success: " + success])
        i += 1

    file.write("\n\nPredictions on Dev set:\n")
    file.write(tabulate(results))

    # Accuracy history plot
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, color='blue', marker='o', label='Training accuracy')
    plt.plot(epochs, val_accuracy, color='red', marker='o', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    print("[DNN] Model accuracy history plot saved in: " + os.path.join(output_dir, "Training_Validation_Accuracy.png"))
    plt.savefig(os.path.join(output_dir, "Training_Validation_Accuracy.png"))
    plt.close()

    # AUC history plot
    train_auc = train.history['AUC']
    val_auc = train.history['val_AUC']
    plt.plot(epochs, train_auc, color='blue', marker='o', label='Training AUC')
    plt.plot(epochs, val_auc, color='red', marker='o', label='Validation AUC')
    plt.title('Training and validation AUC')
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    print("[DNN] Model AUC history plot saved in: " + os.path.join(output_dir, "Training_Validation_AUC.png"))
    plt.savefig(os.path.join(output_dir, "Training_Validation_AUC.png"))
    plt.close()

    # Loss history plot
    loss = train.history['loss']
    val_loss = train.history['val_loss']
    plt.plot(epochs, loss, color='blue', marker='o', label='Training loss')
    plt.plot(epochs, val_loss, color='red', marker='o', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    print("[DNN] Model loss history plot saved in: " + os.path.join(output_dir, "Training_Validation_Loss.png"))
    plt.savefig(os.path.join(output_dir, "Training_Validation_Loss.png"))
    plt.close()

    dnn_data = {}
    # ROC curve and AUC (Dev)
    # roc curve for models
    y_pred_keras = model.predict(X_dev).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_dev, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    dnn_data['fpr_dev'] = fpr_keras
    dnn_data['tpr_dev'] = tpr_keras
    dnn_data['auc_dev'] = auc_keras
    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_dev))]
    p_fpr, p_tpr, _ = roc_curve(y_dev, random_probs, pos_label=1)
    # plot roc curves
    plt.plot(fpr_keras, tpr_keras, color='red',
             label='DNN (AUC = %0.4f)' % auc_keras)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title("ROC curve (Dev), 'alcohol' prediction")
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_alcohol.png"))
    print('[DNN] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_alcohol.png"))
    plt.close()

    print('[DNN] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_DNN_alcohol.txt"))

    return dnn_data


def model_comparison(rf_data, dnn_data, output_dir):
    # ROC curve and AUC comparison (Dev)
    plt.plot([0, 1], [0, 1], 'k--', color='blue')
    plt.plot(rf_data['fpr_dev'], rf_data['tpr_dev'], color='red',
             label='RandomForest (AUC = {:.4f})'.format(rf_data['auc_dev']))
    plt.plot(dnn_data['fpr_dev'], dnn_data['tpr_dev'], color='green',
             label='DNN (AUC = {:.4f})'.format(dnn_data['auc_dev']))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("ROC curve (Dev), 'alcohol' feature")
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "ROCcurveComparison(Dev)_alcohol.png"))
    print('\n[Comparison RF/DNN] Dev ROC curve and AUC comparison saved in: ' + os.path.join(output_dir, "ROCcurveComparison(Dev)_alcohol.png"))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Script to perform 'alcohol' classification in ENSE dataset. Data is splitted in Train(70%), Dev(15%), Test(15%), and missing value imputation, data standardization and class rebalance (undersampling) is performed. Usage example: $python experiments_fuma_ense.py dataENSE2017_m_alcohol.csv -o pathTo/ExperimentsAlcoholm")
    parser.add_argument("input_alcohol_dataset",
                        help="Path to file with input ENSE dataset with 'alcohol' class. For example: 'dataENSE2017_m_alcohol.csv'.")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created Train/Dev/Test sets and results of classification models. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_alcohol_dataset = args['input_alcohol_dataset']
    output_dir = args['output_dir']

    print("\nReading ENSE dataset with 'alcohol' class from: " + str(input_alcohol_dataset))
    dataset_alcohol = pd.read_csv(input_alcohol_dataset, index_col=0)

    print("\nPerforming Train(70%) / Dev(15%) / Test(15%) split of ENSE dataset with 'alcohol' class...")
    Path(os.path.join(output_dir, "Train_Dev_Test")).mkdir(parents=True, exist_ok=True)
    train, dev, test = split_train_dev_test(dataset_alcohol, os.path.join(output_dir, "Train_Dev_Test"))

    Path(os.path.join(output_dir, "Results")).mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(output_dir, "Results")

    print("\nTraining 'alcohol' classification models...")
    Path(os.path.join(output_dir, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "RandomForest")
    rf_data = experiment_rf(train, dev, output_dir_aux)
    Path(os.path.join(output_dir, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "DeepNeuralNetwork")
    dnn_data = experiment_dnn(train, dev, output_dir_aux)
    model_comparison(rf_data, dnn_data, output_dir)

