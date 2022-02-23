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


class PaisnxpPredictor(object):

    @staticmethod
    def randomforest_predictor(dataset, output_dir):
        file = open(os.path.join(output_dir, "REPORT_RandomForest_paisnxp.txt"), "w")
        file.write("'paisnxp' feature prediction:\n\n")

        # Borrar variables clase de la tarea
        dataset = dataset.drop(['pesorec', 'lbw', 'hbw', 'nbw', 'peson', 'premature'], axis=1)

        # Borrar variables que se quieren predecir de los datos 1996-2006
        dataset = dataset.drop(['mimmi', 'fimmi', 'paisnxm', 'estudiom', 'estudiop', 'educm', 'educp', 'conviven'], axis=1)

        # Borrar items con missing values
        dataset = dataset.dropna()

        # Se eliminan los ítems que tienen menos de 3 instancias con esa clase (mínimo 3 para poder dividir en Train/Val/Dev en la DNN)
        counts = dataset['paisnxp'].value_counts()
        dataset = dataset[~dataset['paisnxp'].isin(counts[counts < 3].index)]

        # Train/Dev split
        labels = dataset['paisnxp']
        data = dataset.drop('paisnxp', axis=1)
        indices = dataset.index.tolist()
        X_train, X_dev, y_train, y_dev, indices_train, indices_dev = train_test_split(data, labels, indices,
                                                                                                 test_size=0.2,
                                                                                                 random_state=42,
                                                                                                 shuffle=True,
                                                                                                 stratify=labels)

        X_dev_paisnacp = X_dev['paisnacp'].to_numpy()

        # Borrar ítems de la misma clase que no aparecen en los 2 conjuntos Train/Dev
        clases_borrar = np.setxor1d(np.unique(y_train), np.unique(y_dev))
        X_train['paisnxp'] = y_train
        X_train = X_train[~X_train['paisnxp'].isin(clases_borrar)]
        X_dev['paisnxp'] = y_dev
        X_dev = X_dev[~X_dev['paisnxp'].isin(clases_borrar)]
        y_train = X_train['paisnxp']
        X_train = X_train.drop('paisnxp', 1)
        y_dev = X_dev['paisnxp']
        X_dev = X_dev.drop('paisnxp', 1)

        # Label encoder: convertir códigos de países a categorías desde 0
        encoder = LabelEncoder()
        encoder.fit(y_train)
        mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        mapping_file = open(os.path.join(output_dir, "label_mapping_paisnxp.json"), "w")
        json.dump(mapping, mapping_file)
        mapping_file.close()
        y_train = encoder.transform(y_train)
        y_dev = encoder.transform(y_dev)

        n_classes = len(np.unique(y_train))

        # Predictions
        print("\n[RF] Training RandomForest model...")
        classifier = RandomForestClassifier(max_depth=20)
        classifier.fit(X_train, y_train)
        y_pred_dev = classifier.predict(X_dev)

        # Save the model
        joblib.dump(classifier, os.path.join(output_dir, "RF_MODEL_paisnxp.joblib"), compress=3)
        print('[RF] Trained model saved in: ' + os.path.join(output_dir, "RF_MODEL_paisnxp.joblib"))

        print('\n[RF] Evaluating model on Dev set...')

        # Classification report (Dev)
        y_dev_codes = []
        for y in y_dev:
            y_dev_codes.append(list(mapping.keys())[list(mapping.values()).index(y)])
        y_pred_dev_codes = []
        for y in y_pred_dev:
            y_pred_dev_codes.append(list(mapping.keys())[list(mapping.values()).index(y)])
        file.write("\nClassification Report (Dev):\n" + classification_report(y_dev_codes, y_pred_dev_codes))

        # Dev predictions probabilities
        i = 0
        results = []
        for y in y_dev_codes:
            success = "YES"
            if (y != y_pred_dev_codes[i]):
                success = "NO"
            results.append(["Instance: " + str(indices_dev[i]), "Class: " + str(y),
                            "Predicted: " + str(y_pred_dev_codes[i]),
                            "Success: " + success])
            i += 1
        file.write("\n\nPredictions on Dev set:\n")
        file.write(tabulate(results))

        # Buscar coincidencias entre paisnxm (origen) y paisnacm (nacionalidad) en Test
        mismo_nac_orig = 0
        mismo_nac_orig_pred = 0
        for i in range(len(X_dev_paisnacp)):
            if (X_dev_paisnacp[i] == y_dev_codes[i]):
                mismo_nac_orig += 1
            if (X_dev_paisnacp[i] == y_pred_dev_codes[i]):
                mismo_nac_orig_pred += 1
            i += 1
        prop_mismo_nac_orig = round(mismo_nac_orig / len(X_dev_paisnacp), 4)
        prop_mismo_nac_orig_pred = round(mismo_nac_orig_pred / len(X_dev_paisnacp), 4)
        file.write("\n\nCoincidences between 'paisnxp' (origin) and 'paisnacp' (nationality) in DEV set:\n")
        file.write("\tSame REAL origin ('paisnxp') and nacionality ('paisnacp'): " + str(prop_mismo_nac_orig) + "%\n")
        file.write("\tSame PREDICTED origin ('paisnxp') and nacionality ('paisnacp'): " + str(prop_mismo_nac_orig_pred) + "%\n")

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
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve (Dev), 'paisnxp' feature")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_paisnxp.png"))
        print('[RF] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_paisnxp.png"))
        plt.close()

        print('[RF] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_RandomForest_paisnxp.txt"))

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
        plt.savefig(os.path.join(output_dir, "FeatureImportanceMDI_paisnxp.png"))
        print('[RF] Feature importance based on mean decrease in impurity saved in: ' + os.path.join(output_dir, "FeatureImportanceMDI_paisnxp.png"))
        plt.close()

        return rf_data


    @staticmethod
    def dnn_predictor(dataset, output_dir):
        # Borrar variables clase de la tarea
        dataset = dataset.drop(['pesorec', 'lbw', 'hbw', 'nbw', 'peson', 'premature'], axis=1)

        # Borrar variables que se quieren predecir de los datos 1996-2006
        dataset = dataset.drop(['mimmi', 'fimmi', 'paisnxm', 'estudiom', 'estudiop', 'educm', 'educp', 'conviven'], axis=1)

        # Borrar items con missing values
        dataset = dataset.dropna()

        # Se eliminan los ítems que tienen menos de 3 instancias con esa clase (mínimo 3 para poder dividir en Train/Val/Dev)
        counts = dataset['paisnxp'].value_counts()
        dataset = dataset[~dataset['paisnxp'].isin(counts[counts < 3].index)]

        # Train/Val/Dev split
        labels = dataset['paisnxp']
        data = dataset.drop('paisnxp', axis=1)
        indices = dataset.index.tolist()
        X_train_val, X_dev, y_train_val, y_dev, indices_train_val, indices_dev = train_test_split(data, labels, indices,
                                                                                                 test_size=0.2,
                                                                                                 random_state=42,
                                                                                                 shuffle=True,
                                                                                                 stratify=labels)

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                          test_size=0.2, random_state=42,
                                                          shuffle=True, stratify=y_train_val)

        # Borrar ítems de la misma clase que no aparecen en los 3 conjuntos Train/Val/Dev
        clases_borrar_1 = np.setxor1d(np.unique(y_train), np.unique(y_val))
        clases_borrar_2 = np.setxor1d(np.unique(y_train), np.unique(y_dev))
        clases_borrar_3 = np.setxor1d(np.unique(y_val), np.unique(y_dev))
        clases_borrar = np.concatenate((clases_borrar_1, clases_borrar_2, clases_borrar_3))
        clases_borrar = np.unique(clases_borrar)
        X_train['paisnxp'] = y_train
        X_train = X_train[~X_train['paisnxp'].isin(clases_borrar)]
        X_val['paisnxp'] = y_val
        X_val = X_val[~X_val['paisnxp'].isin(clases_borrar)]
        X_dev['paisnxp'] = y_dev
        X_dev = X_dev[~X_dev['paisnxp'].isin(clases_borrar)]
        y_train = X_train['paisnxp']
        X_train = X_train.drop('paisnxp', 1)
        y_val = X_val['paisnxp']
        X_val = X_val.drop('paisnxp', 1)
        y_dev = X_dev['paisnxp']
        X_dev = X_dev.drop('paisnxp', 1)

        X_dev_paisnacp = X_dev['paisnacp'].to_numpy()

        # Label encoder: convertir códigos de países a categorías desde 0
        encoder = LabelEncoder()
        encoder.fit(y_train)
        mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        mapping_file = open(os.path.join(output_dir, "label_mapping_paisnxp.json"), "w")
        json.dump(mapping, mapping_file)
        mapping_file.close()
        y_train = encoder.transform(y_train)
        y_val = encoder.transform(y_val)
        y_dev = encoder.transform(y_dev)

        # Standardization
        print("\n[DNN] Standardization of datasets...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_dev = scaler.transform(X_dev)
        joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_paisnxp.bin'), compress=True)
        print("[DNN] Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_paisnxp.bin')))

        print("\n[DNN] Training DNN model...")
        input_shape = (len(X_train[0]),)
        n_classes = len(np.unique(y_train))

        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_val)
        y_dev = np_utils.to_categorical(y_dev)

        batch_size = 64
        epochs = 50

        # (input+output)/2
        """model = keras.models.Sequential()
        model.add(keras.layers.Dense(int(len((X_train[0]) + n_classes) / 2), input_shape=input_shape, activation='relu'))
        model.add(keras.layers.Dense(n_classes, activation="softmax"))"""

        # (input*2/3)+output
        """model = keras.models.Sequential()
        model.add(keras.layers.Dense(int((len(X_train[0])*2/3)+n_classes), input_shape=input_shape, activation='relu'))
        model.add(keras.layers.Dense(n_classes, activation="softmax"))"""

        # input*4/3
        """model = keras.models.Sequential()
        model.add(keras.layers.Dense(int(len(X_train[0])*4/3), input_shape=input_shape, activation='relu'))
        model.add(keras.layers.Dense(n_classes, activation="softmax"))"""

        # (input*2/3)+output, (input+output)/2
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(int((len(X_train[0])*2/3)+n_classes), input_shape=input_shape, activation='relu'))
        model.add(keras.layers.Dense(int(len((X_train[0])+n_classes)/2), activation='relu'))
        model.add(keras.layers.Dense(n_classes, activation="softmax"))

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.AUC(name="AUC")])

        file = open(os.path.join(output_dir, "REPORT_DNN_paisnxp.txt"), "w")
        file.write("DNN model: 'paisnxp' feature prediction\n\n")
        model.summary(print_fn=lambda x: file.write(x + '\n'))

        csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
            min_delta=0.0001, cooldown=0, min_lr=0
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

        train = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                          validation_data=(X_val, y_val), callbacks=[csv_logger, reduce_lr, early_stop])

        print('[DNN] Trained model saved in: ' + os.path.join(output_dir, "DNN_MODEL_paisnxp.h5"))
        model.save(os.path.join(output_dir, "DNN_MODEL_paisnxp.h5"))

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
        y_pred_dev_codes = []
        for y in y_pred_dev:
            y_pred_dev_codes.append(list(mapping.keys())[list(mapping.values()).index(y)])
        y_real_dev = y_dev.argmax(axis=-1)
        y_real_dev_codes = []
        for y in y_real_dev:
            y_real_dev_codes.append(list(mapping.keys())[list(mapping.values()).index(y)])
        file.write("\nClassification Report (Dev):\n" + classification_report(y_real_dev_codes, y_pred_dev_codes))

        # Dev predictions probabilities
        i = 0
        results = []
        for y in y_real_dev_codes:
            success = "YES"
            if (y != y_pred_dev_codes[i]):
                success = "NO"
            results.append(["Instance: " + str(indices_dev[i]), "Class: " + str(y),
                            "Predicted: " + str(y_pred_dev_codes[i]),
                            "Success: " + success])
            i += 1
        file.write("\n\nPredictions on Dev set:\n")
        file.write(tabulate(results))

        # Buscar coincidencias entre paisnxm (origen) y paisnacm (nacionalidad) en Test
        mismo_nac_orig = 0
        mismo_nac_orig_pred = 0
        for i in range(len(X_dev_paisnacp)):
            if (X_dev_paisnacp[i] == y_real_dev_codes[i]):
                mismo_nac_orig += 1
            if (X_dev_paisnacp[i] == y_pred_dev_codes[i]):
                mismo_nac_orig_pred += 1
            i += 1
        prop_mismo_nac_orig = round(mismo_nac_orig / len(X_dev_paisnacp), 4)
        prop_mismo_nac_orig_pred = round(mismo_nac_orig_pred / len(X_dev_paisnacp), 4)
        file.write("\n\nCoincidences between 'paisnxp' (origin) and 'paisnacp' (nationality) in DEV set:\n")
        file.write("\tSame REAL origin ('paisnxp') and nacionality ('paisnacp'): " + str(prop_mismo_nac_orig) + "%\n")
        file.write("\tSame PREDICTED origin ('paisnxp') and nacionality ('paisnacp'): " + str(
            prop_mismo_nac_orig_pred) + "%\n")

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
        plt.savefig(os.path.join(output_dir, "Training_Validation_Accuracy.png"))
        print("[DNN] Model accuracy history plot saved in: " + os.path.join(output_dir, "Training_Validation_Accuracy.png"))
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
        plt.savefig(os.path.join(output_dir, "Training_Validation_AUC.png"))
        print("[DNN] Model AUC history plot saved in: " + os.path.join(output_dir, "Training_Validation_AUC.png"))
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
        plt.savefig(os.path.join(output_dir, "Training_Validation_Loss.png"))
        print("[DNN] Model loss history plot saved in: " + os.path.join(output_dir, "Training_Validation_Loss.png"))
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
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve (Dev), 'paisnxp' feature")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_paisnxp.png"))
        print('[DNN] Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_paisnxp.png"))
        plt.close()

        print('[DNN] Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_DNN_paisnxp.txt"))

        return dnn_data


    @staticmethod
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
        plt.title("ROC curve (Dev), 'paisnxp' feature")
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "ROCcurveComparison(Dev)_paisnxp.png"))
        print('\n[Comparison RF/DNN] Dev ROC curve and AUC comparison saved in: ' + os.path.join(output_dir, "ROCcurveComparison(Dev)_paisnxp.png"))
        plt.close()
