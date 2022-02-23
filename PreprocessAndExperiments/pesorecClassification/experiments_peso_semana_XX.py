import argparse
import math
import os
import warnings
from itertools import cycle
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
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


mediana_semanas = 39    # Mediana de semanas de gestación de los datos

def create_dataset_semana_XX(dataset, semana, ecuation):
    indices = dataset.index.values.tolist()
    edad_madre = dataset['edadm'].to_numpy()
    peso_final = dataset['peson'].to_numpy()
    pesorec = dataset['pesorec'].to_numpy()
    proporcion_peso = 0
    if (ecuation == 'China'):
        # Peso % = 500.9 - (51.6 * semana) + (1.727 * (semana ^ 2)) - (0.01718 * (semana ^ 3))
        porcentaje_peso = 500.9 - (51.6 * semana) + (1.727 * math.pow(semana,2)) - (0.01718 * math.pow(semana,3))
        proporcion_peso = porcentaje_peso / 100
    elif (ecuation == 'England'):
        # Peso % = exp(0.578 + 0.332 * semana - 0.00354 * semana ^ 2) / exp(0.578 + 0.332 * mediana_semanas - 0.00354 * mediana_semanas ^ 2)
        proporcion_peso = math.exp(0.578 + 0.332 * semana - 0.00354 * math.pow(semana,2)) / math.exp(0.578 + 0.332 * mediana_semanas - 0.00354 * math.pow(mediana_semanas,2))
    peso_semana_XX = np.round(peso_final * proporcion_peso, 4)
    df = pd.DataFrame({'edadm': edad_madre, 'peso_semana_' + str(semana): peso_semana_XX, 'pesorec': pesorec}, index=indices)
    return df

def create_dataset_semana_XX_socioeconomico(dataset, semana, ecuation):
    dataset_socioeconomico = dataset[['edadm','edadm6','edadm35','edadp','edadp35','mforeign','fforeign','paisnacm','paisnacp','mimmi','fimmi','paisnxm','paisnxp','estudiom','estudiop','educm','educp','profm','profp','occupm','occupp','casada','ecivm','pareja','conviven']]
    peso_final = dataset['peson'].to_numpy()
    pesorec = dataset['pesorec'].to_numpy()
    proporcion_peso = 0
    if (ecuation == 'China'):
        # Peso % = 500.9 - (51.6 * semana) + (1.727 * (semana ^ 2)) - (0.01718 * (semana ^ 3))
        porcentaje_peso = 500.9 - (51.6 * semana) + (1.727 * math.pow(semana, 2)) - (0.01718 * math.pow(semana, 3))
        proporcion_peso = porcentaje_peso / 100
    elif (ecuation == 'England'):
        # Peso % = exp(0.578 + 0.332 * semana - 0.00354 * semana ^ 2) / exp(0.578 + 0.332 * mediana_semanas - 0.00354 * mediana_semanas ^ 2) * 100
        proporcion_peso = math.exp(0.578 + 0.332 * semana - 0.00354 * math.pow(semana, 2)) / math.exp(0.578 + 0.332 * mediana_semanas - 0.00354 * math.pow(mediana_semanas, 2))
    peso_semana_XX = np.round(peso_final * proporcion_peso, 4)
    dataset_socioeconomico['peso_semana_' + str(semana)] = peso_semana_XX
    dataset_socioeconomico['pesorec'] = pesorec
    return dataset_socioeconomico


def split_train_dev_test(dataset, output_dir):
    # Train(70%) / Dev(15%) / Test(15%)
    labels = dataset['pesorec']
    data = dataset.drop('pesorec', axis=1)
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(data, labels,
                                                                test_size=0.15, random_state=42,
                                                                shuffle=True, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev,
                                                      test_size=(len(y_test) / len(y_train_dev)),
                                                      random_state=42, shuffle=True,
                                                      stratify=y_train_dev)

    X_train['pesorec'] = y_train
    X_dev['pesorec'] = y_dev
    X_test['pesorec'] = y_test

    X_train.to_csv(os.path.join(output_dir, "train.csv"))
    print("Train set saved in : " + os.path.join(output_dir, "train.csv"))
    X_dev.to_csv(os.path.join(output_dir, "dev.csv"))
    print("Dev set saved in : " + os.path.join(output_dir, "dev.csv"))
    X_test.to_csv(os.path.join(output_dir, "test.csv"))
    print("Test set saved in : " + os.path.join(output_dir, "test.csv"))


def experiment_rf(train, dev, output_dir):
    file = open(os.path.join(output_dir, "REPORT_RandomForest_pesorec.txt"), "w")
    file.write("RandomForest model: 'pesorec' prediction:\n\n")

    # Features/Labels
    y_train = train['pesorec']
    X_train = train.drop('pesorec', axis=1)
    feature_names = X_train.columns
    y_dev = dev['pesorec']
    X_dev = dev.drop('pesorec', axis=1)
    indices_dev = dev.index.tolist()

    # Predictions
    print("\n[RF] Training RandomForest model...")
    n_classes = 3
    classifier = RandomForestClassifier(max_depth=25)
    classifier.fit(X_train, y_train)
    y_pred_dev = classifier.predict(X_dev)

    # Save the model
    joblib.dump(classifier, os.path.join(output_dir, "RF_MODEL_pesorec.joblib"), compress=3)
    print('[RF] Trained model saved in: ' + os.path.join(output_dir, "RF_MODEL_pesorec.joblib"))

    print('\n[RF] Evaluating model...')

    # Classification report (Dev)
    y_pred_dev_names = []
    for val in y_pred_dev:
        if (val == 0):
            y_pred_dev_names.append('LOW')
        elif (val == 1):
            y_pred_dev_names.append('NORMAL')
        elif (val == 2):
            y_pred_dev_names.append('HIGH')
    y_real_dev_names = []
    for val in y_dev:
        if (val == 0):
            y_real_dev_names.append('LOW')
        elif (val == 1):
            y_real_dev_names.append('NORMAL')
        elif (val == 2):
            y_real_dev_names.append('HIGH')
    file.write("\nClassification Report (Dev):\n" + classification_report(y_dev, y_pred_dev))

    # Confusion matrix (Dev)
    conf_mat_dev = confusion_matrix(y_dev, y_pred_dev)
    fp = conf_mat_dev.sum(axis=0) - np.diag(conf_mat_dev)
    fn = conf_mat_dev.sum(axis=1) - np.diag(conf_mat_dev)
    tp = np.diag(conf_mat_dev)
    tn = conf_mat_dev.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)
    specificity_dev = tn / (tn + fp)
    sensitivity_dev = tp / (tp + fn)
    file.write("\nspecificity\t\t" + str(specificity_dev))
    file.write("\nsensitivity\t\t" + str(sensitivity_dev) + "\n\n")
    df_cm_dev = pd.DataFrame(conf_mat_dev, index=['LOW', 'NORMAL', 'HIGH'], columns=['LOW', 'NORMAL', 'HIGH'])
    sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix (Dev), 'pesorec' prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfusionMatrix_pesorec.png"))
    plt.close()
    print('[RF] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_pesorec.png"))

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

    rf_data = {}
    # ROC curve and AUC (Dev)
    # Compute ROC curve and ROC area for each class
    predictions_dev = classifier.predict_proba(X_dev)
    y_dev = np_utils.to_categorical(y_dev)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_dev[:, i], predictions_dev[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_dev.ravel(), predictions_dev.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    rf_data['fpr_micro_dev'] = fpr["micro"]
    rf_data['tpr_micro_dev'] = tpr["micro"]
    rf_data['auc_micro_dev'] = roc_auc["micro"]
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    rf_data['fpr_macro_dev'] = fpr["macro"]
    rf_data['tpr_macro_dev'] = tpr["macro"]
    rf_data['auc_macro_dev'] = roc_auc["macro"]
    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='RandomForest, micro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='RandomForest, macro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='RandomForest, ROC curve of class {0} (AUC = {1:0.4f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve (Dev), 'pesorec' prediction")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_pesorec.png"))
    print('[RF] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_pesorec.png"))
    plt.close()

    print('[RF] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_RandomForest_pesorec.txt"))

    # Feature importance (Mean decrease in impurity)
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "FeatureImportanceMDI_pesorec.png"))
    print('[RF] Feature importance based on mean decrease in impurity saved in: ' + os.path.join(output_dir, "FeatureImportanceMDI_pesorec.png"))
    plt.close()

    return rf_data


def experiment_dnn(train, dev, output_dir):
    # Features/Labels
    y_train_val = train['pesorec']
    X_train_val = train.drop('pesorec', axis=1)
    y_dev = dev['pesorec']
    X_dev = dev.drop('pesorec', axis=1)
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
    joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_pesorec.bin'), compress=True)
    print("[DNN] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_pesorec.bin')))

    print("\n[DNN] Training DNN model...")
    input_shape = (n_features,)
    n_classes = 3

    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    y_dev = np_utils.to_categorical(y_dev)

    batch_size = 64
    epochs = 50

    # (input+output)/2
    """model = keras.models.Sequential()
    model.add(keras.layers.Dense(int(n_features + n_classes) / 2), input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))"""

    # (input*2/3)+output
    """model = keras.models.Sequential()
    model.add(keras.layers.Dense(int((n_features*2/3)+n_classes), input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))"""

    # input*4/3
    """model = keras.models.Sequential()
    model.add(keras.layers.Dense(int(n_features*4/3), input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))"""

    # (input*2/3)+output, (input+output)/2
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(int((n_features * 2 / 3) + n_classes), input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(int((n_features + n_classes) / 2), activation='relu'))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.AUC(name="AUC")])

    file = open(os.path.join(output_dir, "REPORT_DNN_pesorec.txt"), "w")
    file.write("DNN model: 'pesorec' prediction\n\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))

    csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

    train = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                      validation_data=(X_val, y_val), callbacks=[csv_logger, reduce_lr, early_stop])

    print('[DNN] Trained model saved in: ' + os.path.join(output_dir, "DNN_MODEL_pesorec.h5"))
    model.save(os.path.join(output_dir, "DNN_MODEL_pesorec.h5"))

    # DEV evaluation
    print('\n[DNN] Evaluating model on Dev set...')
    dev_eval = model.evaluate(X_dev, y_dev, verbose=1)
    file.write("\n\nDev dataset evaluation:\n")
    file.write('\tDev loss: ' + str(round(dev_eval[0], 4)) + "\n")
    file.write('\tDev accuracy: ' + str(round(dev_eval[1], 4)) + "\n")
    file.write('\tDev recall: ' + str(round(dev_eval[2], 4)) + "\n")
    file.write('\tDev precision: ' + str(round(dev_eval[3], 4)) + "\n")
    file.write('\tDev AUC: ' + str(round(dev_eval[4], 4)) + "\n\n")

    # Classification report (Dev)
    predictions_dev = model.predict(X_dev)
    y_pred_dev = predictions_dev.argmax(axis=-1)
    y_real_dev = y_dev.argmax(axis=-1)
    y_pred_dev_names = []
    for val in y_pred_dev:
        if (val == 0):
            y_pred_dev_names.append('LOW')
        elif (val == 1):
            y_pred_dev_names.append('NORMAL')
        elif (val == 2):
            y_pred_dev_names.append('HIGH')
    y_real_dev_names = []
    for val in y_real_dev:
        if (val == 0):
            y_real_dev_names.append('LOW')
        elif (val == 1):
            y_real_dev_names.append('NORMAL')
        elif (val == 2):
            y_real_dev_names.append('HIGH')
    file.write("\nClassification Report (Dev):\n" + classification_report(y_real_dev_names, y_pred_dev_names))

    # Confusion matrix (Dev)
    conf_mat_dev = confusion_matrix(y_real_dev, y_pred_dev)
    fp = conf_mat_dev.sum(axis=0) - np.diag(conf_mat_dev)
    fn = conf_mat_dev.sum(axis=1) - np.diag(conf_mat_dev)
    tp = np.diag(conf_mat_dev)
    tn = conf_mat_dev.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)
    specificity_dev = tn / (tn + fp)
    sensitivity_dev = tp / (tp + fn)
    specificity_dev_print = []
    specificity_dev_print.append("LOW: " + str(specificity_dev[0]))
    specificity_dev_print.append("NORMAL: " + str(specificity_dev[1]))
    specificity_dev_print.append("HIGH: " + str(specificity_dev[2]))
    sensitivity_dev_print = []
    sensitivity_dev_print.append("LOW: " + str(sensitivity_dev[0]))
    sensitivity_dev_print.append("NORMAL: " + str(sensitivity_dev[1]))
    sensitivity_dev_print.append("HIGH: " + str(sensitivity_dev[2]))
    file.write("\nspecificity\t\t" + str(specificity_dev_print))
    file.write("\nsensitivity\t\t" + str(sensitivity_dev_print) + "\n\n")
    df_cm_dev = pd.DataFrame(conf_mat_dev, index=['LOW', 'NORMAL', 'HIGH'], columns=['LOW', 'NORMAL', 'HIGH'])
    sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix (Dev), 'pesorec' prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfusionMatrix_pesorec.png"))
    plt.close()
    print('[DNN] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_pesorec.png"))

    # Dev predictions probabilities
    i = 0
    results = []
    for y in y_real_dev_names:
        success = "YES"
        if (y != y_pred_dev_names[i]):
            success = "NO"
        results.append(["Instance: " + str(indices_dev[i]), "Class: " + str(y),
                        "Predicted: " + str(y_pred_dev_names[i]),
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
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_dev[:, i], predictions_dev[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_dev.ravel(), predictions_dev.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    dnn_data['fpr_micro_dev'] = fpr["micro"]
    dnn_data['tpr_micro_dev'] = tpr["micro"]
    dnn_data['auc_micro_dev'] = roc_auc["micro"]
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    dnn_data['fpr_macro_dev'] = fpr["macro"]
    dnn_data['tpr_macro_dev'] = tpr["macro"]
    dnn_data['auc_macro_dev'] = roc_auc["macro"]
    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='DNN, micro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='DNN, macro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        if (i == 0):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='DNN, ROC curve of class LOW (AUC = {0:0.4f})' ''.format(roc_auc[i]))
        elif (i == 1):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='DNN, ROC curve of class NORMAL (AUC = {0:0.4f})' ''.format(roc_auc[i]))
        elif (i == 2):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='DNN, ROC curve of class HIGH (AUC = {0:0.4f})' ''.format(roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve (Dev), 'pesorec' prediction")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_pesorec.png"))
    print('[DNN] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_pesorec.png"))
    plt.close()

    print('[DNN] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_DNN_pesorec.txt"))

    return dnn_data


def model_comparison(rf_data, dnn_data, output_dir):
    # ROC curve and AUC comparison (Dev)
    plt.plot([0, 1], [0, 1], 'k--', color='blue')
    plt.plot(rf_data['fpr_macro_dev'], rf_data['tpr_macro_dev'], color='red',
             label='RandomForest, macro-average ROC curve (AUC = {:.4f})'.format(rf_data['auc_macro_dev']))
    plt.plot(dnn_data['fpr_macro_dev'], dnn_data['tpr_macro_dev'], color='green',
             label='DNN, macro-average ROC curve (AUC = {:.4f})'.format(dnn_data['auc_macro_dev']))
    plt.plot(rf_data['fpr_micro_dev'], rf_data['tpr_micro_dev'], color='red', linestyle=':',
             label='RandomForest, micro-average ROC curve (AUC = {:.4f})'.format(rf_data['auc_micro_dev']))
    plt.plot(dnn_data['fpr_micro_dev'], dnn_data['tpr_micro_dev'], color='green', linestyle=':',
             label='DNN, micro-average ROC curve (AUC = {:.4f})'.format(dnn_data['auc_micro_dev']))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("ROC curve (Dev), 'pesorec' prediction")
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "ROCcurveComparison(Dev)_pesorec.png"))
    print('\n[Comparison RF/DNN] Dev ROC curve and AUC comparison saved in: ' + os.path.join(output_dir, "ROCcurveComparison(Dev)_pesorec.png"))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to perform 'pesorec' classification experiments adding 'peso_semana_XX' fake feature. This feature estimates the weight of the fetus in the XX week of pregnancy. Week for weight estimation can be selected and two estimation ecuations are available: the first tested over Chinese population and valid for 22-40 weeks of gestation, and the second tested over the English population and valid for 10-42 weeks of gestation. Two models are trained: a baseline only using mom's age and 'peso_semana_XX' feature, and a model adding all socioeconomic features. Data is splitted in Train(70%), Dev(15%), Test(15%). Usage example: $python experiments_peso_semana_XX.py dataPerinatal_predictions.csv -w 32 -e China -o pathTo/PesoSemanaXXExperiments")
    parser.add_argument("input_pesorec_dataset",
                        help="Path to file with input dataset for 'pesorec' classification. For example: 'dataPerinatal_predictions.csv'.")
    parser.add_argument("-w", "--week",
                        help="Week for calculating weight estimation. Default option: 32",
                        default=32)
    parser.add_argument("-e", "--ecuation",
                        help="Ecuation for calculating weight estimation: China population ecuation [China] or England population ecuation [England]. Default option: [China]",
                        default='China')
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created datasets with 'peson_semanas_XX' fake features, Train/Dev/Test sets and results of classification models. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_pesorec_dataset = args['input_pesorec_dataset']
    semana = int(args['week'])
    ecuation = args['ecuation']
    output_dir = args['output_dir']

    if ecuation not in ['China', 'England']:
        parser.error("'ecuation' value must be [China] or [England].")

    print("\nReading dataset for 'pesorec' classification from: " + str(input_pesorec_dataset))
    dataset_pesorec = pd.read_csv(input_pesorec_dataset, index_col=0)

    Path(os.path.join(output_dir, "Week_" + str(semana) + "_" + str(ecuation))).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "Week_" + str(semana) + "_" + str(ecuation))

    print("\nCreating dataset for baseline with 'edadm' and 'peso_semana_" + str(semana) + "' features...")
    Path(os.path.join(output_dir_aux, "Baseline")).mkdir(parents=True, exist_ok=True)
    dataset_pesorec_baseline = create_dataset_semana_XX(dataset_pesorec, semana, ecuation)
    print("Saving created dataset in: " + str(os.path.join(output_dir_aux, "Baseline", os.path.basename(input_pesorec_dataset).split('.')[0] + "_baseline(peso_semana_" + str(semana) + "_" + str(ecuation) + ").csv")))
    dataset_pesorec_baseline.to_csv(os.path.join(output_dir_aux, "Baseline", os.path.basename(input_pesorec_dataset).split('.')[0] + "_baseline(peso_semana_" + str(semana) + "_" + str(ecuation) + ").csv"))

    print("\nCreating dataset for model with all socioeconomic features...")
    Path(os.path.join(output_dir_aux, "Socioeconomic")).mkdir(parents=True, exist_ok=True)
    dataset_pesorec_socioeconomic = create_dataset_semana_XX_socioeconomico(dataset_pesorec, semana, ecuation)
    print("Saving created dataset in: " + str(os.path.join(output_dir_aux, "Socioeconomic", os.path.basename(input_pesorec_dataset).split('.')[0] + "_socioeconomic(peso_semana_" + str(semana) + "_" + str(ecuation) + ").csv")))
    dataset_pesorec_socioeconomic.to_csv(os.path.join(output_dir_aux, "Socioeconomic", os.path.basename(input_pesorec_dataset).split('.')[0] + "_socioeconomic(peso_semana_" + str(semana) + "_" + str(ecuation) + ").csv"))

    print("\nPerforming Train(70%) / Dev(15%) / Test(15%) split of '" + str(os.path.basename(input_pesorec_dataset).split('.')[0] + "_baseline(peso_semana_" + str(semana) + "_" + str(ecuation) + ").csv") + "' dataset...")
    Path(os.path.join(output_dir_aux, "Baseline", "Train_Dev_Test")).mkdir(parents=True, exist_ok=True)
    split_train_dev_test(dataset_pesorec_baseline, os.path.join(output_dir_aux, "Baseline", "Train_Dev_Test"))

    print("\nPerforming Train(70%) / Dev(15%) / Test(15%) split of '" + str(os.path.basename(input_pesorec_dataset).split('.')[0] + "_socioeconomic(peso_semana_" + str(semana) + "_" + str(ecuation) + ").csv") + "' dataset...")
    Path(os.path.join(output_dir_aux, "Socioeconomic", "Train_Dev_Test")).mkdir(parents=True, exist_ok=True)
    split_train_dev_test(dataset_pesorec_socioeconomic, os.path.join(output_dir_aux, "Socioeconomic", "Train_Dev_Test"))

    # Baseline 'pesorec' classification models
    print("\nTraining baseline 'pesorec' classification models...")
    Path(os.path.join(output_dir_aux, "Baseline", "Results")).mkdir(parents=True, exist_ok=True)
    print('Reading Train dataset from: ' + os.path.join(output_dir_aux, "Baseline", "Train_Dev_Test", "train.csv"))
    train = pd.read_csv(os.path.join(output_dir_aux, "Baseline", "Train_Dev_Test", "train.csv"), index_col=0)
    print('Reading Dev dataset from: ' + os.path.join(output_dir_aux, "Baseline", "Train_Dev_Test", "dev.csv"))
    dev = pd.read_csv(os.path.join(output_dir_aux, "Baseline", "Train_Dev_Test", "dev.csv"), index_col=0)
    Path(os.path.join(output_dir_aux, "Baseline", "Results", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_rf = os.path.join(output_dir_aux, "Baseline", "Results", "RandomForest")
    rf_data = experiment_rf(train, dev, output_dir_aux_rf)
    Path(os.path.join(output_dir_aux, "Baseline", "Results", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_dnn = os.path.join(output_dir_aux, "Baseline", "Results", "DeepNeuralNetwork")
    dnn_data = experiment_dnn(train, dev, output_dir_aux_dnn)
    model_comparison(rf_data, dnn_data, os.path.join(output_dir_aux, "Baseline", "Results"))

    # Socioeconomic 'pesorec' classification models
    print("\nTraining socioeconomic 'pesorec' classification models...")
    Path(os.path.join(output_dir_aux, "Socioeconomic", "Results")).mkdir(parents=True, exist_ok=True)
    print('Reading Train dataset from: ' + os.path.join(output_dir_aux, "Socioeconomic", "Train_Dev_Test", "train.csv"))
    train = pd.read_csv(os.path.join(output_dir_aux, "Socioeconomic", "Train_Dev_Test", "train.csv"), index_col=0)
    print('Reading Dev dataset from: ' + os.path.join(output_dir_aux, "Socioeconomic", "Train_Dev_Test", "dev.csv"))
    dev = pd.read_csv(os.path.join(output_dir_aux, "Socioeconomic", "Train_Dev_Test", "dev.csv"), index_col=0)
    Path(os.path.join(output_dir_aux, "Socioeconomic", "Results", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_rf = os.path.join(output_dir_aux, "Socioeconomic", "Results", "RandomForest")
    rf_data = experiment_rf(train, dev, output_dir_aux_rf)
    Path(os.path.join(output_dir_aux, "Socioeconomic", "Results", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_dnn = os.path.join(output_dir_aux, "Socioeconomic", "Results", "DeepNeuralNetwork")
    dnn_data = experiment_dnn(train, dev, output_dir_aux_dnn)
    model_comparison(rf_data, dnn_data, os.path.join(output_dir_aux, "Socioeconomic", "Results"))


