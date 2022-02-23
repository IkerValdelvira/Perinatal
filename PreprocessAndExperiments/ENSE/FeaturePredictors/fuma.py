import json
import os
import pickle
from itertools import cycle

import joblib
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from numpy import interp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class fumaPredictor(object):

    @staticmethod
    def create_dataset(dataset_ENSE, dataset_compatible):
        dataset_compatible_copy = dataset_compatible.copy()
        indices = dataset_compatible_copy.index.values.tolist()
        v121_values = dataset_ENSE["V121"].loc[indices]
        dataset_compatible_copy['fuma'] = v121_values
        dataset_compatible_copy = dataset_compatible_copy.dropna()
        dataset_compatible_copy = dataset_compatible_copy[~dataset_compatible_copy['fuma'].isin([8,9])]
        dataset_compatible_copy['fuma'] = dataset_compatible_copy['fuma'].replace([3,4], 0)
        dataset_compatible_copy['fuma'] = dataset_compatible_copy['fuma'].replace([1, 2], 1)
        return dataset_compatible_copy


    @staticmethod
    def randomforest_predictor(dataset, output_dir):
        file = open(os.path.join(output_dir, "REPORT_RandomForest_fuma.txt"), "w")
        file.write("'fuma' feature prediction:\n\n")

        # Train/Dev/Test split
        labels = dataset['fuma']
        data = dataset.drop('fuma', axis=1)
        indices = dataset.index.tolist()
        X_train, X_dev, y_train, y_dev, indices_train, indices_dev = train_test_split(data, labels, indices,
                                                                                      test_size=0.2,
                                                                                      random_state=42,
                                                                                      shuffle=True,
                                                                                      stratify=labels)

        # Predictions
        print("\n[RF] Training RandomForest model...")
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train);
        y_pred_dev = classifier.predict(X_dev)

        print('\n[RF] Evaluating model on Dev set...')

        # Classification report (Dev)
        file.write("\nClassification Report (Dev):\n" + classification_report(y_dev, y_pred_dev))

        # Confusion matrix (Dev)
        conf_mat_dev = confusion_matrix(y_dev, y_pred_dev)
        tn, fp, fn, tp = conf_mat_dev.ravel()
        specificity_dev = tn / (tn + fp)
        sensitivity_dev = tp / (tp + fn)
        file.write("\nspecificity\t\t" + str(specificity_dev))
        file.write("\nsensitivity\t\t" + str(sensitivity_dev) + "\n\n")
        df_cm_dev = pd.DataFrame(conf_mat_dev, index=['0', '1'], columns=['0', '1'])
        sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix (Dev), 'fuma' feature")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ConfusionMatrix_fuma.png"))
        plt.close()
        print('[RF] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_fuma.png"))

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
        # roc curve for models
        y_pred_prob = classifier.predict_proba(X_dev)
        fpr, tpr, thresh = roc_curve(y_dev, y_pred_prob[:, 1], pos_label=1)
        auc_metric = auc(fpr, tpr)
        rf_data['fpr_dev'] = fpr
        rf_data['tpr_dev'] = tpr
        rf_data['auc_dev'] = auc_metric
        # roc curve for tpr = fpr
        random_probs = [0 for i in range(len(y_dev))]
        p_fpr, p_tpr, _ = roc_curve(y_dev, random_probs, pos_label=1)
        # plot roc curves
        plt.plot(fpr, tpr, color='red',
                 label='Random Forest (AUC = %0.4f)' % auc_metric)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title("ROC curve (Dev), 'fuma' feature")
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_fuma.png"))
        print('[RF] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_fuma.png"))
        plt.close()

        print('[RF] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_RandomForest_fuma.txt"))

        # Feature importance (Mean decrease in impurity)
        feature_names = X_train.columns
        importances = classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=feature_names)
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "FeatureImportanceMDI_fuma.png"))
        print('[RF] Feature importance based on mean decrease in impurity saved in: ' + os.path.join(output_dir, "FeatureImportanceMDI_fuma.png"))
        plt.close()

        classifier_final = RandomForestClassifier()
        classifier_final.fit(data, labels)

        joblib.dump(classifier_final, os.path.join(output_dir, "RF_MODEL_fuma.joblib"), compress=3)
        print('[RF] Trained model saved in: ' + os.path.join(output_dir, "RF_MODEL_fuma.joblib"))

        return rf_data


    @staticmethod
    def dnn_predictor(dataset, output_dir):
        # Train/Val/Dev split
        labels = dataset['fuma']
        data = dataset.drop('fuma', axis=1)
        indices = dataset.index.tolist()
        X_train_val, X_dev, y_train_val, y_dev, indices_train_val, indices_dev = train_test_split(data, labels, indices,
                                                                                                  test_size=0.2,
                                                                                                  random_state=42,
                                                                                                  shuffle=True,
                                                                                                  stratify=labels)
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

        print("\n[DNN] Training DNN model...")
        input_shape = (len(X_train[0]),)

        batch_size = 64
        epochs = 50

        # (input*2/3)+output, (input+output)/2
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(int((len(X_train[0]) * 2 / 3) + 1), input_shape=input_shape, activation='relu'))
        model.add(keras.layers.Dense(int(len((X_train[0]) + 1) / 2), activation='relu'))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.AUC(name="AUC")])

        file = open(os.path.join(output_dir, "REPORT_DNN_fuma.txt"), "w")
        file.write("DNN model: 'fuma' feature prediction\n\n")
        model.summary(print_fn=lambda x: file.write(x + '\n'))

        csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
            min_delta=0.0001, cooldown=0, min_lr=0
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

        train = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                          validation_data=(X_val, y_val), callbacks=[csv_logger, reduce_lr, early_stop])

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
        df_cm_dev = pd.DataFrame(conf_mat_dev, index=['0', '1'], columns=['0', '1'])
        sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix (Dev), 'fuma' feature")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ConfusionMatrix_fuma.png"))
        plt.close()
        print('[DNN] Confusion matrix saved in: ' + os.path.join(output_dir, "ConfusionMatrix_fuma.png"))

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
        plt.title("ROC curve (Dev), 'fuma' feature")
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_fuma.png"))
        print('[DNN] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_fuma.png"))
        plt.close()

        print('[DNN] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_DNN_fuma.txt"))

        print("\n[DNN] FINAL model: train + dev")
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels,
                                                          test_size=0.2, random_state=42,
                                                          shuffle=True, stratify=labels)

        # Standardization
        print("\n[DNN] Standardization of datasets...")
        scaler = StandardScaler()
        scaler.fit(data_train)
        data_train = scaler.transform(data_train)
        data_val = scaler.transform(data_val)
        joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_fuma.bin'), compress=True)
        print("[DNN] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_fuma.bin')))

        batch_size = 64
        epochs = 50

        # (input*2/3)+output, (input+output)/2
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(int((len(X_train[0]) * 2 / 3) + 1), input_shape=input_shape, activation='relu'))
        model.add(keras.layers.Dense(int(len((X_train[0]) + 1) / 2), activation='relu'))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.AUC(name="AUC")])

        model.fit(data_train, labels_train, batch_size=batch_size, epochs=epochs, verbose=1,
                  validation_data=(data_val, labels_val), callbacks=[reduce_lr, early_stop])

        print('[DNN] Trained FINAL model saved in: ' + os.path.join(output_dir, "DNN_MODEL_fuma.h5"))
        model.save(os.path.join(output_dir, "DNN_MODEL_fuma.h5"))

        return dnn_data


    @staticmethod
    def model_comparison(rf_data, dnn_data, output_dir):
        # ROC curve and AUC comparison (Dev)
        plt.plot([0, 1], [0, 1], 'k--', color='blue')
        plt.plot(rf_data['fpr_dev'], rf_data['tpr_dev'], color='red',
                 label='RandomForest (AUC = {:.4f})'.format(rf_data['auc_dev']))
        plt.plot(dnn_data['fpr_dev'], dnn_data['tpr_dev'], color='green',
                 label='DNN (AUC = {:.4f})'.format(dnn_data['auc_dev']))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title("ROC curve (Dev), 'fuma' feature")
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "ROCcurveComparison(Dev)_fuma.png"))
        print('\n[Comparison RF/DNN] Dev ROC curve and AUC comparison saved in: ' + os.path.join(output_dir, "ROCcurveComparison(Dev)_fuma.png"))
        plt.close()