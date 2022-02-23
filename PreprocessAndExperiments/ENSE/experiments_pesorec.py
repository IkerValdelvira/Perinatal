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
    # Solo se va a usar la clase 'pesorec' --> Borrar 'lbw', 'nbw', 'hbw', 'peson' y 'premature'
    dataset = dataset.drop(['lbw', 'nbw', 'hbw', 'peson', 'premature'], 1)

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

    return X_train, X_dev, X_test


def experiment_rf(train, dev, resample, output_dir):
    file = open(os.path.join(output_dir, "REPORT_RandomForest_pesorec.txt"), "w")
    file.write("RandomForest model: 'pesorec' prediction:\n\n")

    # Features/Labels
    y_train = train['pesorec']
    X_train = train.drop('pesorec', axis=1)
    feature_names = X_train.columns
    y_dev = dev['pesorec']
    X_dev = dev.drop('pesorec', axis=1)
    indices_dev = dev.index.tolist()

    # Resampling TRAIN set
    if (resample == 'OS'):  # Oversampling
        print("\n[RF] Oversampling not majority classes ('LOW' and 'HIGH') of training set...")
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    if (resample == 'SMOTE'):  # Oversampling with SMOTE
        print("\n[RF] Oversampling minority class of training set with SMOTE...")
        oversampler = SMOTE(sampling_strategy='minority')
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    elif (resample == 'US'):  # Undersampling
        print("\n[RF] Undersampling not minority classes ('NORMAL' and 'HIGH') of training set...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    elif (resample == 'TomekLinks'):  # Undersampling with TomekLinks
        print("\n[RF] Undersampling mayority class of training set with TomekLinks...")
        undersampler = TomekLinks(sampling_strategy='majority')
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        indices_selected = undersampler.sample_indices_
        print(indices_selected)

    elif (resample == 'OUS'):  # Oversampling(0.1)/Undersampling(0.5)
        print("\n[RF] Oversampling not majority classes ('LOW' and 'HIGH') of training set by 10%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        # oversampler = SMOTE(random_state=42, sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]), 2: int(occurrences[2] * 1.1)})
        oversampler = RandomOverSampler(random_state=42,
                                        sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]),
                                                           2: int(occurrences[2] * 1.1)})
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        print("[RF] Undersampling majority class ('NORMAL') of training set by 50%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        undersampler = RandomUnderSampler(random_state=42,
                                          sampling_strategy={0: int(occurrences[0]), 1: int(occurrences[1] * 0.5),
                                                             2: int(occurrences[2])})
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    # print(np.asarray((unique, counts)).T)

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
    file.write("\nClassification Report (Dev):\n" + classification_report(y_real_dev_names, y_pred_dev_names))

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


def experiment_dnn(train, dev, resample, output_dir):
    # Features/Labels
    y_train = train['pesorec']
    X_train = train.drop('pesorec', axis=1)
    indices_train = train.index.tolist()
    column_names = X_train.columns
    y_dev = dev['pesorec']
    X_dev = dev.drop('pesorec', axis=1)
    indices_dev = dev.index.tolist()
    n_features = len(X_train.columns)

    # Standardization
    print("\n[DNN] Standardization of datasets...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)
    joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_pesorec.bin'), compress=True)
    print("[DNN] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_pesorec.bin')))

    # Resampling TRAIN set
    if (resample == 'OS'):  # Oversampling
        print("\n[DNN] Oversampling not majority classes ('LOW' and 'HIGH') of training set...")
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    if (resample == 'SMOTE'):  # Oversampling with SMOTE
        print("\n[DNN] Oversampling minority class of training set with SMOTE...")
        oversampler = SMOTE(sampling_strategy='minority')
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    elif (resample == 'US'):  # Undersampling
        print("\n[DNN] Undersampling not minority classes ('NORMAL' and 'HIGH') of training set...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    elif (resample == 'TomekLinks'):  # Undersampling with TomekLinks
        print("\n[DNN] Undersampling mayority class of training set with TomekLinks...")
        undersampler = TomekLinks(sampling_strategy='majority')
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        indices_selected = undersampler.sample_indices_
        print(indices_selected)

    elif (resample == 'OUS'):  # Oversampling(0.1)/Undersampling(0.5)
        print("\n[DNN] Oversampling not majority classes ('LOW' and 'HIGH') of training set by 10%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        # oversampler = SMOTE(random_state=42, sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]), 2: int(occurrences[2] * 1.1)})
        oversampler = RandomOverSampler(random_state=42,
                                        sampling_strategy={0: int(occurrences[0] * 1.1), 1: int(occurrences[1]),
                                                           2: int(occurrences[2] * 1.1)})
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        print("[DNN] Undersampling majority class ('NORMAL') of training set by 50%...")
        unique, counts = np.unique(y_train, return_counts=True)
        occurrences = dict(zip(unique, counts))
        undersampler = RandomUnderSampler(random_state=42,
                                          sampling_strategy={0: int(occurrences[0]), 1: int(occurrences[1] * 0.5),
                                                             2: int(occurrences[2])})
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    print("\n[DNN] Training DNN model...")
    input_shape = (n_features,)
    n_classes = 3

    y_train = np_utils.to_categorical(y_train)
    y_dev = np_utils.to_categorical(y_dev)

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
                      validation_split=0.2, callbacks=[csv_logger, reduce_lr, early_stop])

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
    parser = argparse.ArgumentParser( description="Script to perform 'pesorec' classification using RandomForest and optimized DNN on Perinatal dataset with ENSE features added. Data is splitted in Train(70%), Dev(15%), Test(15%), and multiple oversampling/undersampling techniques can be selected.. Usage example: $python experiments_pesorec.py dataPerinatal_predictions_ENSE.csv -o pathTo/ExperimentsPesorec")
    parser.add_argument("input_perinatal_dataset",
                        help="Path to file with input Perinatal dataset with ENSE features added. For example: 'dataPerinatal_predictions_ENSE.csv'.")
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

    print("\nReading Perinatal dataset with ENSE features for 'pesorec' classification from: " + str(input_perinatal_dataset))
    dataset_pesorec = pd.read_csv(input_perinatal_dataset, index_col=0)

    print("\nPerforming Train(70%) / Dev(15%) / Test(15%) split of Perinatal dataset with ENSE features added...")
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

    print("\nTraining 'pesorec' classification models...")
    Path(os.path.join(output_dir, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "RandomForest")
    rf_data = experiment_rf(train, dev, resample, output_dir_aux)
    Path(os.path.join(output_dir, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "DeepNeuralNetwork")
    dnn_data = experiment_dnn(train, dev, resample, output_dir_aux)
    model_comparison(rf_data, dnn_data, output_dir)

