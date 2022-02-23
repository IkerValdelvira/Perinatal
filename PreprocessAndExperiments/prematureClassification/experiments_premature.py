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

def split_train_dev_test(dataset, output_dir):
    # Solo se va a usar la clase 'premature' --> Borrar 'lbw', 'nbw', 'hbw', 'peson' y 'pesorec'
    dataset = dataset.drop(['lbw', 'nbw', 'hbw', 'peson', 'pesorec'], 1)

    # Train(70%) / Dev(15%) / Test(15%)
    labels = dataset['premature']
    data = dataset.drop('premature', axis=1)
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(data, labels,
                                                                test_size=0.15, random_state=42,
                                                                shuffle=True, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev,
                                                      test_size=(len(y_test) / len(y_train_dev)),
                                                      random_state=42, shuffle=True,
                                                      stratify=y_train_dev)

    X_train['premature'] = y_train
    X_dev['premature'] = y_dev
    X_test['premature'] = y_test

    X_train.to_csv(os.path.join(output_dir, "train.csv"))
    print("Train set saved in : " + os.path.join(output_dir, "train.csv"))
    X_dev.to_csv(os.path.join(output_dir, "dev.csv"))
    print("Dev set saved in : " + os.path.join(output_dir, "dev.csv"))
    X_test.to_csv(os.path.join(output_dir, "test.csv"))
    print("Test set saved in : " + os.path.join(output_dir, "test.csv"))

    return X_train, X_dev, X_test


def experiment_rf(train, dev, resample, output_dir):
    file = open(os.path.join(output_dir, "REPORT_RandomForest_premature.txt"), "w")
    file.write("RandomForest model: 'premature' prediction:\n\n")

    # Features/Labels
    y_train = train['premature']
    X_train = train.drop('premature', axis=1)
    feature_names = X_train.columns
    y_dev = dev['premature']
    X_dev = dev.drop('premature', axis=1)
    indices_dev = dev.index.tolist()

    # Resampling TRAIN set
    if (resample == 'OS'):  # Oversampling
        print("\n[RF] Oversampling minority class of training set...")
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    if (resample == 'SMOTE'):  # Oversampling with SMOTE
        print("\n[RF] Oversampling minority class of training set with SMOTE...")
        oversampler = SMOTE(sampling_strategy='minority')
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    elif (resample == 'US'):  # Undersampling
        print("\n[RF] Undersampling majority class of training set...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    elif (resample == 'TomekLinks'):  # Undersampling with TomekLinks
        print("\n[RF] Undersampling majority class of training set with TomekLinks...")
        undersampler = TomekLinks(sampling_strategy='majority')
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        indices_selected = undersampler.sample_indices_
        print(indices_selected)

    elif (resample == 'OUS'):  # Oversampling(0.1)/Undersampling(0.5)
        print("\n[RF] Oversampling minority class of training set by 10%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        #oversampler = SMOTE(random_state=42, sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]), 2: int(occurrences[2] * 1.1)})
        oversampler = RandomOverSampler(random_state=42, sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]), 2: int(occurrences[2] * 1.1)})
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        print("[RF] Undersampling majority class of training set by 50%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        undersampler = RandomUnderSampler(random_state=42, sampling_strategy={0: int(occurrences[0]), 1: int(occurrences[1] * 0.5), 2: int(occurrences[2])})
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    #print(np.asarray((unique, counts)).T)

    # Predictions
    print("\n[RF] Training RandomForest model...")
    classifier = RandomForestClassifier(max_depth=25)
    classifier.fit(X_train, y_train)
    y_pred_dev = classifier.predict(X_dev)

    # Save the model
    joblib.dump(classifier, os.path.join(output_dir, "RF_MODEL_premature.joblib"), compress=3)
    print('[RF] Trained model saved in: ' + os.path.join(output_dir, "RF_MODEL_premature.joblib"))

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
    plt.title("Confusion matrix (Dev), 'premature' prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfusionMatrix_premature.png"))
    plt.close()
    print('[RF] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_premature.png"))

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
    fpr, tpr, thresholds = roc_curve(y_dev, predictions_dev[:,1], pos_label=1)
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
    plt.title("ROC curve (Dev), 'premature' prediction")
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_premature.png"))
    print('[RF] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_premature.png"))
    plt.close()

    print('[RF] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_RandomForest_premature.txt"))

    # Feature importance (Mean decrease in impurity)
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "FeatureImportanceMDI_premature.png"))
    print('[RF] Feature importance based on mean decrease in impurity saved in: ' + os.path.join(output_dir, "FeatureImportanceMDI_premature.png"))
    plt.close()

    return rf_data


def experiment_dnn(train, dev, resample, output_dir):
    # Features/Labels
    y_train_val = train['premature']
    X_train_val = train.drop('premature', axis=1)
    y_dev = dev['premature']
    X_dev = dev.drop('premature', axis=1)
    indices_dev = dev.index.tolist()
    n_features = len(X_train_val.columns)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.2, random_state=42,
                                                      shuffle=True, stratify=y_train_val)

    # Standardization
    print("\n[DNN] Standardization of datasets...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_dev = scaler.transform(X_dev)
    joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_premature.bin'), compress=True)
    print("[DNN] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_premature.bin')))

    # Resampling TRAIN set
    if (resample == 'OS'):  # Oversampling
        print("\n[DNN] Oversampling minority class of training set...")
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    if (resample == 'SMOTE'):  # Oversampling with SMOTE
        print("\n[DNN] Oversampling minority class of training set with SMOTE...")
        oversampler = SMOTE(sampling_strategy='minority')
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    elif (resample == 'US'):  # Undersampling
        print("\n[DNN] Undersampling majority class of training set...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    elif (resample == 'TomekLinks'):  # Undersampling with TomekLinks
        print("\n[DNN] Undersampling mayority class of training set with TomekLinks...")
        undersampler = TomekLinks(sampling_strategy='majority')
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        indices_selected = undersampler.sample_indices_
        print(indices_selected)

    elif (resample == 'OUS'):  # Oversampling(0.1)/Undersampling(0.5)
        print("\n[DNN] Oversampling minority class of training set by 10%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        #oversampler = SMOTE(random_state=42, sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]), 2: int(occurrences[2] * 1.1)})
        oversampler = RandomOverSampler(random_state=42, sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]), 2: int(occurrences[2] * 1.1)})
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        print("[DNN] Undersampling majority class of training set by 50%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        undersampler = RandomUnderSampler(random_state=42, sampling_strategy={0: int(occurrences[0]), 1: int(occurrences[1] * 0.5), 2: int(occurrences[2])})
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    #print(np.asarray((unique, counts)).T)

    print("\n[DNN] Training DNN model...")
    input_shape = (n_features,)

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

    file = open(os.path.join(output_dir, "REPORT_DNN_premature.txt"), "w")
    file.write("DNN model: 'premature' prediction\n\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))

    csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

    train = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                      validation_data=(X_val, y_val), callbacks=[csv_logger, reduce_lr, early_stop])

    print('[DNN] Trained model saved in: ' + os.path.join(output_dir, "DNN_MODEL_premature.h5"))
    model.save(os.path.join(output_dir, "DNN_MODEL_premature.h5"))

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
    plt.title("Confusion matrix (Dev), 'premature' prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfusionMatrix_premature.png"))
    plt.close()
    print('[DNN] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_premature.png"))

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
    plt.title("ROC curve (Dev), 'premature' prediction")
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_premature.png"))
    print('[DNN] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_premature.png"))
    plt.close()

    print('[DNN] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_DNN_premature.txt"))

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
    plt.title("ROC curve (Dev), 'premature' feature")
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "ROCcurveComparison(Dev)_premature.png"))
    print('\n[Comparison RF/DNN] Dev ROC curve and AUC comparison saved in: ' + os.path.join(output_dir, "ROCcurveComparison(Dev)_premature.png"))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to perform 'premature' classification using RandomForest and DNN on Perinatal dataset. Data is splitted in Train(70%), Dev(15%), Test(15%), and multiple oversampling/undersampling techniques can be selected. Usage example: $python experiments_premature.py dataPerinatal_predictions.csv -o pathTo/ExperimentsPremature")
    parser.add_argument("input_perinatal_dataset",
                        help="Path to file with input Perinatal dataset. For example: 'dataPerinatal_predictions.csv'.")
    parser.add_argument("-rs", "--resample_method",
                        help="Method to resample data of training set: oversampling not majority classes ('LOW' and 'HIGH') [OS], undersampling not minority classes ('LOW' and 'NORMAL') [US] or oversampling not majority classes ('LOW' and 'HIGH') by a 10 percent and undersampling majority class ('NORMAL') by a 50 percent [OUS], oversampling minority class using SMOTE [SMOTE] or undersampling majority class using TomekLinks [TomekLinks]. Default option: empty.",
                        default='')
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created Train/Dev/Test sets and results of classification models. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_perinatal_dataset = args['input_perinatal_dataset']
    resample = args['resample_method']
    output_dir = args['output_dir']

    if resample not in ['OS', 'US', 'OUS', 'SMOTE', 'TomekLinks', '']:
        parser.error("'resample_method' value must be [OS], [US], [OUS], [SMOTE], [TomekLinks] or empty.")

    print("\nReading Perinatal dataset for 'premature' classification from: " + str(input_perinatal_dataset))
    dataset_pesorec = pd.read_csv(input_perinatal_dataset, index_col=0)

    print("\nPerforming Train(70%) / Dev(15%) / Test(15%) split of Perinatal dataset...")
    Path(os.path.join(output_dir, "Train_Dev_Test")).mkdir(parents=True, exist_ok=True)
    train, dev, test = split_train_dev_test(dataset_pesorec, os.path.join(output_dir, "Train_Dev_Test"))

    if (resample == 'OS'):
        Path(os.path.join(output_dir, "Results_Oversampling")).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, "Results_Oversampling")
    elif (resample == 'US'):
        Path(os.path.join(output_dir, "Results_Undersampling")).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, "Results_Undersampling")
    elif (resample == 'OUS'):
        Path(os.path.join(output_dir, "Results_Oversampling_Undersampling")).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, "Results_Oversampling_Undersampling")
    elif (resample == 'SMOTE'):
        Path(os.path.join(output_dir, "Results_SMOTE")).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, "Results_SMOTE")
    elif (resample == 'TomekLinks'):
        Path(os.path.join(output_dir, "Results_TomekLinks")).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, "Results_TomekLinks")
    else:
        Path(os.path.join(output_dir, "Results")).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, "Results")

    print("\nTraining 'premature' classification models...")
    Path(os.path.join(output_dir, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "RandomForest")
    rf_data = experiment_rf(train, dev, resample, output_dir_aux)
    Path(os.path.join(output_dir, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "DeepNeuralNetwork")
    dnn_data = experiment_dnn(train, dev, resample, output_dir_aux)
    model_comparison(rf_data, dnn_data, output_dir)

