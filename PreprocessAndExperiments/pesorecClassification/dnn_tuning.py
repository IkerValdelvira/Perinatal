import argparse
import os
import warnings
from itertools import cycle
from pathlib import Path
import shutil
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from keras_tuner import HyperModel
from numpy import interp
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tabulate import tabulate
from tensorflow import keras
import keras_tuner

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def experiment_dnn(train, dev, output_dir):
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

    print("\n[DNN] DNN model hyperparameter tuning...")
    input_shape = (n_features,)
    n_classes = 3

    y_train = np_utils.to_categorical(y_train)
    y_dev = np_utils.to_categorical(y_dev)

    # Model with hyperparameter tuning
    class DNNHyperModel(HyperModel):
        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes

        def build(self, hp):
            model = keras.models.Sequential()
            model.add(keras.Input(shape=self.input_shape))
            for i in range(hp.Int('layers', min_value=1, max_value=6, step=1)):
                model.add(keras.layers.Dense(
                    units=hp.Int('units_layer' + str(i), min_value=32, max_value=512, step=32),
                    activation=hp.Choice('dense_activation_layer' + str(i), values=['relu', 'tanh', 'sigmoid'])
                ))
            model.add(keras.layers.Dense(self.num_classes, activation="softmax"))

            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                          optimizer=tf.keras.optimizers.Adam(hp.Float(
                              'learning_rate',
                              min_value=1e-4,
                              max_value=1e-2,
                              sampling='LOG',
                              default=1e-3
                          )),
                          metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                                   tf.keras.metrics.Recall(name='recall'),
                                   tf.keras.metrics.Precision(name='precision'),
                                   tf.keras.metrics.AUC(name="AUC")])
            return model

    hypermodel = DNNHyperModel(input_shape=input_shape, num_classes=n_classes)

    file = open(os.path.join(output_dir, "REPORT_DNN_pesorec.txt"), "w")
    file.write("DNN model: 'pesorec' prediction\n\n")

    batch_size = 64
    epochs = 50

    csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

    tuner = keras_tuner.tuners.BayesianOptimization(
        hypermodel,
        objective='val_AUC',
        max_trials=20,
        seed=42,
        directory='bayesian_optimization',
        project_name='tuner_dnn_pesorec',
        overwrite=True)

    tuner.search_space_summary()

    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[reduce_lr, early_stop])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    num_layers = best_hps.get("layers")
    print("\n[DNN] optimal number of hidden layers: " + str(num_layers))
    for i in range(num_layers):
        print("[DNN] optimal number of units in layer " + str(i) + ": " + str(best_hps.get("units_layer" + str(i))))
        print("[DNN] optimal activation function in layer " + str(i) + ": " + str(best_hps.get("dense_activation_layer" + str(i))))
    print("[DNN] optimal learning rate: {:.8f}".format(best_hps.get("learning_rate")))

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write("\nOptimal number of hidden layers: " + str(num_layers))
    for i in range(num_layers):
        file.write("\nOptimal number of units in layer " + str(i) + ": " + str(best_hps.get("units_layer" + str(i))))
        file.write("\nOptimal activation function in layer " + str(i) + ": " + str(best_hps.get("dense_activation_layer" + str(i))))
    file.write("\nOptimal learning rate: {:.8f}".format(best_hps.get("learning_rate")))

    # Build and train best model
    model = tuner.hypermodel.build(best_hps)
    train = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[csv_logger, reduce_lr, early_stop])

    print('\n[DNN] Trained model saved in: ' + os.path.join(output_dir, "DNN_MODEL_pesorec.h5"))
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

    # Move folder with tuner information
    original = 'bayesian_optimization'
    target = os.path.join(output_dir)
    shutil.move(original, target)



if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Script to perform DNN hyperparameter optimization in 'pesorec' classification. Following hyperparameters are tuned: number of hidden layers, number of neurons in hidden layers, activation function in each layer, and initial learning rate. Bayesian optimization is used for hyperparameter tuning. Usage example: $python dnn_tunnig.py train_predictions.csv dev.csv -o pathTo/DNNTuning")
    parser.add_argument("input_pesorec_train", help="Path to file with input Train dataset for 'pesorec' classification. For example: 'train_predictions.csv'.")
    parser.add_argument("input_pesorec_dev", help="Path to file with input Dev dataset for 'pesorec' classification. For example: 'dev.csv'.")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created classification model, evaluation and hyperparameter tuning report. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_pesorec_train = args['input_pesorec_train']
    input_pesorec_dev = args['input_pesorec_dev']
    output_dir = args['output_dir']

    print("\nReading Train dataset for 'pesorec' classification from: " + str(input_pesorec_train))
    train_pesorec = pd.read_csv(input_pesorec_train, index_col=0)

    print("\nReading Dev dataset for 'pesorec' classification from: " + str(input_pesorec_dev))
    dev_pesorec = pd.read_csv(input_pesorec_dev, index_col=0)

    print("\nPerforming DNN model training with hyperparameter optimization...")
    Path(os.path.join(output_dir, "Results")).mkdir(parents=True, exist_ok=True)
    experiment_dnn(train_pesorec, dev_pesorec, os.path.join(output_dir, "Results"))

