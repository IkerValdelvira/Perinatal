import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

from PreprocessAndExperiments.FeaturePredictors.conviven import ConvivenPredictor
from PreprocessAndExperiments.FeaturePredictors.educm import EducmPredictor
from PreprocessAndExperiments.FeaturePredictors.educp import EducpPredictor
from PreprocessAndExperiments.FeaturePredictors.estudiom import EstudiomPredictor
from PreprocessAndExperiments.FeaturePredictors.estudiop import EstudiopPredictor
from PreprocessAndExperiments.FeaturePredictors.fimmi import FimmiPredictor
from PreprocessAndExperiments.FeaturePredictors.mimmi import MimmiPredictor
from PreprocessAndExperiments.FeaturePredictors.paisnxm import PaisnxmPredictor
from PreprocessAndExperiments.FeaturePredictors.paisnxp import PaisnxpPredictor

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def impute_missing_values(dataset):
    Path(os.path.join(output_dir, "ModelsFeaturePredictors")).mkdir(parents=True, exist_ok=True)

    # MimmiPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "mimmi")).mkdir(parents=True, exist_ok=True)
    print("\nTraining 'mimmi' feature predictor model...")
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "mimmi", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "mimmi", "RandomForest")
    rf_data = MimmiPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "mimmi", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "mimmi", "DeepNeuralNetwork")
    dnn_data = MimmiPredictor.dnn_predictor(dataset, output_dir_aux)
    MimmiPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # FimmiPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "fimmi")).mkdir(parents=True, exist_ok=True)
    print("\nTraining 'fimmi' feature predictor model...")
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "fimmi", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "fimmi", "RandomForest")
    rf_data = FimmiPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "fimmi", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "fimmi", "DeepNeuralNetwork")
    dnn_data = FimmiPredictor.dnn_predictor(dataset, output_dir_aux)
    FimmiPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # EducmPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "educm")).mkdir(parents=True, exist_ok=True)
    print("\nTraining 'educm' feature predictor model...")
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "RandomForest")
    rf_data = EducmPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "DeepNeuralNetwork")
    dnn_data = EducmPredictor.dnn_predictor(dataset, output_dir_aux)
    EducmPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # EducpPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "educm")).mkdir(parents=True, exist_ok=True)
    print("\nTraining 'educp' feature predictor model...")
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "RandomForest")
    rf_data = EducpPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "DeepNeuralNetwork")
    dnn_data = EducpPredictor.dnn_predictor(dataset, output_dir_aux)
    EducpPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # EstudiomPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiom")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiom", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "estudiom", "RandomForest")
    rf_data = EstudiomPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiom", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "estudiom", "DeepNeuralNetwork")
    dnn_data = EstudiomPredictor.dnn_predictor(dataset, output_dir_aux)
    EstudiomPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # EstudiopPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiop")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiop", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "estudiop", "RandomForest")
    rf_data = EstudiopPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiop", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "estudiop", "DeepNeuralNetwork")
    dnn_data = EstudiopPredictor.dnn_predictor(dataset, output_dir_aux)
    EstudiopPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # ConvivenPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "conviven")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "conviven", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "conviven", "RandomForest")
    rf_data = ConvivenPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "conviven", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "conviven", "DeepNeuralNetwork")
    dnn_data = ConvivenPredictor.dnn_predictor(dataset, output_dir_aux)
    ConvivenPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # PaisnxmPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxm")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxm", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxm", "RandomForest")
    rf_data = PaisnxmPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxm", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxm", "DeepNeuralNetwork")
    dnn_data = PaisnxmPredictor.dnn_predictor(dataset, output_dir_aux)
    PaisnxmPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # PaisnxpPredictor
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxp")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxp", "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxp", "RandomForest")
    rf_data = PaisnxpPredictor.randomforest_predictor(dataset, output_dir_aux)
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxp", "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux = os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxp", "DeepNeuralNetwork")
    dnn_data = PaisnxpPredictor.dnn_predictor(dataset, output_dir_aux)
    PaisnxpPredictor.model_comparison(rf_data, dnn_data, output_dir)

    # Predecir los missing values con los modelos entrenados
    model = load_model(os.path.join(output_dir, "ModelsFeaturePredictors", "mimmi", "DeepNeuralNetwork", "DNN_MODEL_mimmi.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "mimmi", "DeepNeuralNetwork"), model, 'mimmi')

    model = load_model(os.path.join(output_dir, "ModelsFeaturePredictors", "fimmi", "DeepNeuralNetwork", "DNN_MODEL_fimmi.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "fimmi", "DeepNeuralNetwork"), model, 'fimmi')

    model = load_model(os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "DeepNeuralNetwork", "DNN_MODEL_educm.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "educm", "DeepNeuralNetwork"), model, 'educm')

    model = load_model( os.path.join(output_dir, "ModelsFeaturePredictors", "educp", "DeepNeuralNetwork", "DNN_MODEL_educp.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "educp", "DeepNeuralNetwork"), model, 'educp')

    model = load_model(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiom", "DeepNeuralNetwork", "DNN_MODEL_estudiom.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "estudiom", "DeepNeuralNetwork"), model, 'estudiom')

    model = load_model(os.path.join(output_dir, "ModelsFeaturePredictors", "estudiop", "DeepNeuralNetwork", "DNN_MODEL_estudiop.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "estudiop", "DeepNeuralNetwork"), model, 'estudiop')

    model = load_model(os.path.join(output_dir, "ModelsFeaturePredictors", "conviven", "DeepNeuralNetwork", "DNN_MODEL_conviven.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "conviven", "DeepNeuralNetwork"), model, 'conviven')

    model = load_model( os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxm", "DeepNeuralNetwork", "DNN_MODEL_paisnxm.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxm", "DeepNeuralNetwork"), model, 'paisnxm')

    model = load_model(os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxp", "DeepNeuralNetwork", "DNN_MODEL_paisnxp.h5"))
    dataset = predictions(dataset, os.path.join(output_dir, "ModelsFeaturePredictors", "paisnxp", "DeepNeuralNetwork"), model, 'paisnxp')

    return dataset


def predictions(dataset, model_folder, model, feature):
    # Borrar variables clase de la tarea
    dataset_copy = dataset.drop(['premature', 'peson', 'lbw', 'nbw', 'hbw', 'pesorec'], axis=1)

    # Borrar variables con missing values en los datos 1996-2006
    if (feature == 'mimmi'):
        dataset_copy = dataset_copy.drop(['fimmi', 'paisnxm', 'paisnxp', 'estudiom', 'estudiop', 'educm', 'educp', 'conviven'], axis=1)
    elif (feature == 'fimmi'):
        dataset_copy = dataset_copy.drop(['mimmi', 'paisnxm', 'paisnxp', 'estudiom', 'estudiop', 'educm', 'educp', 'conviven'], axis=1)
    elif (feature == 'paisnxm'):
        dataset_copy = dataset_copy.drop(['mimmi', 'fimmi', 'paisnxp', 'estudiom', 'estudiop', 'educm', 'educp', 'conviven'], axis=1)
    elif (feature == 'paisnxp'):
        dataset_copy = dataset_copy.drop(['mimmi', 'fimmi', 'paisnxm', 'estudiom', 'estudiop', 'educm', 'educp', 'conviven'], axis=1)
    elif (feature == 'estudiom'):
        dataset_copy = dataset_copy.drop(['mimmi', 'fimmi', 'paisnxm', 'paisnxp', 'estudiop', 'educm', 'educp', 'conviven'], axis=1)
    elif (feature == 'estudiop'):
        dataset_copy = dataset_copy.drop(['mimmi', 'fimmi', 'paisnxm', 'paisnxp', 'estudiom', 'educm', 'educp', 'conviven'], axis=1)
    elif (feature == 'educm'):
        dataset_copy = dataset_copy.drop(['mimmi', 'fimmi', 'paisnxm', 'paisnxp', 'estudiom', 'estudiop', 'educp', 'conviven'], axis=1)
    elif (feature == 'educp'):
        dataset_copy = dataset_copy.drop(['mimmi', 'fimmi', 'paisnxm', 'paisnxp', 'estudiom', 'estudiop', 'educm', 'conviven'], axis=1)
    elif (feature == 'conviven'):
        dataset_copy = dataset_copy.drop(['mimmi', 'fimmi', 'paisnxm', 'paisnxp', 'estudiom', 'estudiop', 'educm', 'educp'], axis=1)

    # Seleccionar los items con NaN en la feature a predecir
    pred_dataset = dataset_copy[dataset_copy[feature].isnull()]

    # Eliminar items que tienen missing values en cualquiera de las features menos la que se quiere predecir
    pred_dataset = pred_dataset.drop(feature, 1)
    pred_dataset = pred_dataset.dropna()

    # Predecir la feature
    print("\nPredicting values of feature '" + feature + "' in items that is possible...")
    scaler = joblib.load(os.path.join(model_folder, "std_scaler_" + feature + ".bin"))
    pred_dataset_scaled = scaler.transform(pred_dataset)
    if (feature in ['mimmi', 'fimmi', 'conviven']):     # Features binarias
        pred_feature = np.where(model.predict(pred_dataset_scaled) > 0.5, 1, 0)
        pred_feature = [item for sublist in pred_feature for item in sublist]
    else:   # Features categóricas
        pred_feature = model.predict(pred_dataset_scaled).argmax(axis=-1)
        if (feature in ['estudiom', 'estudiop']):   # El modelo clasifica de 0-11 y las categorías son de 1-12
            pred_feature = [x + 1 for x in pred_feature]
        elif (feature in ['paisnxm', 'paisnxp']):   # El modelo clasifica desde el valor 0 y hay que mapear las categorías
            label_mapping_file = open(os.path.join(model_folder, "label_mapping_" + feature + ".json"), "r")
            label_mapping = label_mapping_file.read()
            label_mapping = json.loads(label_mapping)
            pred_feature_codes = []
            for pred in pred_feature:
                code = list(label_mapping.keys())[list(label_mapping.values()).index(pred)]
                pred_feature_codes.append(code)
            pred_feature = pred_feature_codes

    pred_dataset[feature] = pred_feature
    dataset.loc[pred_dataset.index.values.tolist(), feature] = pred_dataset[feature]

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to preprocess Perinatal dataset. 3 dataset will be created: 'dataPerinatal_remove_items.csv' removing all items with missing values, 'dataPerinatal_remove_features.csv' removing features with missing values, and 'dataPerinatal_predicted.csv' predicting missing values. Usage example: $python dataset_preprocessing.py dataPerinatal_converted.csv -o pathTo/Preprocess")
    parser.add_argument("input_dataset", help="Path to converted Perinatal dataset: 'dataPerinatal_converted.csv'.")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for created datasets. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_dataset = args['input_dataset']
    output_dir = args['output_dir']

    print('\nReading converted Perinatal dataset from: ' + str(input_dataset))
    dataset = pd.read_csv(input_dataset)

    # Seleccionar items sin missing values exceptuando las variables mimmi, fimmi, educm, educp, estudiom, estudiop, paisnxm, paisnxp y conviven
    dataset = dataset[['numhv', 'firstborn', 'singleton', 'propar', 'mespar', 'anopar', 'sexo', 'edadm', 'edadm6', 'edadm35', 'edadp', 'edadp35', 'mforeign', 'fforeign', 'paisnacm', 'paisnacp', 'mimmi', 'fimmi', 'paisnxm', 'paisnxp', 'estudiom', 'estudiop', 'educm', 'educp', 'profm', 'profp', 'occupm', 'occupp', 'casada', 'ecivm', 'pareja', 'conviven', 'pesorec', 'lbw', 'hbw', 'nbw', 'peson', 'premature']]
    dataset = dataset.dropna(subset=['numhv', 'firstborn', 'singleton', 'propar', 'mespar', 'anopar', 'sexo', 'edadm', 'edadm6', 'edadm35', 'edadp', 'edadp35', 'mforeign', 'fforeign', 'paisnacm', 'paisnacp', 'profm', 'profp', 'occupm', 'occupp', 'casada', 'ecivm', 'pareja', 'pesorec'])

    print("\nCreating 'dataPerinatal_remove_items.csv' dataset with items without missing values...")
    dataset_remove_items = dataset.dropna()

    print("\nCreating 'dataPerinatal_remove_features.csv' dataset removing features with missing values...")
    dataset_remove_features = dataset.dropna(axis=1)

    print("\nCreating 'dataPerinatal_remove_features.csv' dataset imputing predicted missing values in some features ('mimmi', 'fimmi', 'educm', 'educp', 'estudiom', 'estudiop', 'paisnxm', 'paisnxp' and 'conviven')...")
    dataset_predictions = impute_missing_values(dataset)

    print("\nSaving 'dataPerinatal_remove_items.csv' dataset in: " + str(os.path.join(output_dir, "dataPerinatal_remove_items.csv")))
    dataset_remove_items.to_csv(os.path.join(output_dir, "dataPerinatal_remove_items.csv"))

    print("\nSaving 'dataPerinatal_remove_features.csv' dataset in: " + str(os.path.join(output_dir, "dataPerinatal_remove_features.csv")))
    dataset_remove_features.to_csv(os.path.join(output_dir, "dataPerinatal_remove_features.csv"))

    print("\nSaving 'dataPerinatal_predictions.csv' dataset in: " + str(os.path.join(output_dir, "dataPerinatal_predictions.csv")))
    dataset_predictions.to_csv(os.path.join(output_dir, "dataPerinatal_predictions.csv"))

