import argparse
import json
import os
from pathlib import Path

import joblib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import seaborn as sns

from FeaturePredictors.alcohol import alcoholPredictor
from FeaturePredictors.fuma import fumaPredictor

def distribution_V121(dataset, output_dir):
    labels = ['1', '2', '3', '4']
    classes = pd.value_counts(dataset['V121'], sort=True, normalize=True) * 100
    barlist = plt.bar(classes.index, classes.values, alpha=0.5)
    for i, index in enumerate(classes.index):
        if (index == 1):
            barlist[i].set_color('red')
        elif (index == 2):
            barlist[i].set_color('orange')
        elif (index == 3):
            barlist[i].set_color('yellow')
        elif (index == 4):
            barlist[i].set_color('green')
    plt.title("V121 feature distribution")
    plt.xticks(range(1,5), labels)
    plt.xlabel("V121")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Distribution_V121.png"))
    print("\nHistogram of feature distribution 'V121' saved in: " + os.path.join(output_dir, "Distribution_V121.png"))
    plt.close()

def distribution_W127(dataset, output_dir):
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    classes = pd.value_counts(dataset['W127'], sort=True, normalize=True) * 100
    barlist = plt.bar(classes.index, classes.values, alpha=0.5)
    for i, index in enumerate(classes.index):
        if (index == 1):
            barlist[i].set_color('black')
        elif (index == 2):
            barlist[i].set_color('red')
        elif (index == 3):
            barlist[i].set_color('orange')
        elif (index == 4):
            barlist[i].set_color('yellow')
        elif (index == 5):
            barlist[i].set_color('green')
        elif (index == 6):
            barlist[i].set_color('blue')
        elif (index == 7):
            barlist[i].set_color('cyan')
        elif (index == 8):
            barlist[i].set_color('purple')
        elif (index == 9):
            barlist[i].set_color('pink')
    plt.title("W127 feature distribution")
    plt.xticks(range(1, 10), labels)
    plt.xlabel("W127")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Distribution_W127.png"))
    print("\nHistogram of feature distribution 'W127' saved in: " + os.path.join(output_dir, "Distribution_W127.png"))
    plt.close()


def distribution_fuma_m(dataset, output_dir):
    labels = ['0', '1']
    classes = pd.value_counts(dataset['fuma'], sort=True, normalize=True) * 100
    barlist = plt.bar(classes.index, classes.values, alpha=0.5)
    for i, index in enumerate(classes.index):
        if (index == 0):
            barlist[i].set_color('green')
        elif (index == 1):
            barlist[i].set_color('red')
    plt.title("Women 'fuma' feature distribution")
    plt.xticks(range(2), labels)
    plt.xlabel("fuma")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Distribution_fuma_m.png"))
    print("\nHistogram of feature distribution 'fuma' for women saved in: " + os.path.join(output_dir, "Distribution_fuma_m.png"))
    plt.close()

def distribution_fuma_p(dataset, output_dir):
    labels = ['0', '1']
    classes = pd.value_counts(dataset['fuma'], sort=True, normalize=True) * 100
    barlist = plt.bar(classes.index, classes.values, alpha=0.5)
    for i, index in enumerate(classes.index):
        if (index == 0):
            barlist[i].set_color('green')
        elif (index == 1):
            barlist[i].set_color('red')
    plt.title("Men 'fuma' feature distribution")
    plt.xticks(range(2), labels)
    plt.xlabel("fuma")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Distribution_fuma_p.png"))
    print("\nHistogram of feature distribution 'fuma' for men saved in: " + os.path.join(output_dir, "Distribution_fuma_p.png"))
    plt.close()


def distribution_alcohol_m(dataset, output_dir):
    labels = ['0', '1']
    classes = pd.value_counts(dataset['alcohol'], sort=True, normalize=True) * 100
    barlist = plt.bar(classes.index, classes.values, alpha=0.5)
    for i, index in enumerate(classes.index):
        if (index == 0):
            barlist[i].set_color('green')
        elif (index == 1):
            barlist[i].set_color('red')
    plt.title("Women 'alcohol' feature distribution")
    plt.xticks(range(2), labels)
    plt.xlabel("alcohol")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Distribution_alcohol_m.png"))
    print("\nHistogram of feature distribution 'alcohol' for women saved in: " + os.path.join(output_dir, "Distribution_alcohol_m.png"))
    plt.close()


def distribution_alcohol_p(dataset, output_dir):
    labels = ['0', '1']
    classes = pd.value_counts(dataset['alcohol'], sort=True, normalize=True) * 100
    barlist = plt.bar(classes.index, classes.values, alpha=0.5)
    for i, index in enumerate(classes.index):
        if (index == 0):
            barlist[i].set_color('green')
        elif (index == 1):
            barlist[i].set_color('red')
    plt.title("Men 'alcohol' feature distribution")
    plt.xticks(range(2), labels)
    plt.xlabel("alcohol")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Distribution_alcohol_p.png"))
    print("\nHistogram of feature distribution 'alcohol' for men saved in: " + os.path.join(output_dir, "Distribution_alcohol_p.png"))
    plt.close()


def predictions(dataset, model_folder, model, feature):
    pesorec = dataset['pesorec']
    dataset = dataset.drop('pesorec', 1)

    # Cambio de 'propar' a 'ccaapar'
    if 'ccaapar' not in dataset.columns:
        dataset['propar'] = dataset['propar'].replace([4,11,14,18,21,23,29,41], 1)
        dataset['propar'] = dataset['propar'].replace([22,44,50], 2)
        dataset['propar'] = dataset['propar'].replace([33], 3)
        dataset['propar'] = dataset['propar'].replace([7], 4)
        dataset['propar'] = dataset['propar'].replace([35,38], 5)
        dataset['propar'] = dataset['propar'].replace([39], 6)
        dataset['propar'] = dataset['propar'].replace([5,9,24,35,37,40,42,47,49], 7)
        dataset['propar'] = dataset['propar'].replace([2,13,16,19,45], 8)
        dataset['propar'] = dataset['propar'].replace([8,17,25,43], 9)
        dataset['propar'] = dataset['propar'].replace([3,12,46], 10)
        dataset['propar'] = dataset['propar'].replace([6,10], 11)
        dataset['propar'] = dataset['propar'].replace([15,27,32,36], 12)
        dataset['propar'] = dataset['propar'].replace([28], 13)
        dataset['propar'] = dataset['propar'].replace([30], 14)
        dataset['propar'] = dataset['propar'].replace([31], 15)
        dataset['propar'] = dataset['propar'].replace([1,20,48], 16)
        dataset['propar'] = dataset['propar'].replace([26], 17)
        dataset['propar'] = dataset['propar'].replace([51], 18)
        dataset['propar'] = dataset['propar'].replace([52], 19)
        dataset = dataset.rename(columns={'propar': 'ccaapar'})

    madre_padre = feature[-1]
    pred_dataset = dataset.copy()
    if (madre_padre == 'm'):
        pred_dataset = dataset[['ccaapar', 'edadm', 'mforeign', 'mimmi', 'estudiom', 'profm', 'casada', 'ecivm', 'conviven']]
    elif (madre_padre == 'p'):
        pred_dataset = dataset[['ccaapar', 'edadp', 'fforeign', 'fimmi', 'estudiop', 'profp', 'casada', 'ecivm', 'conviven']]

    # Predecir la feature
    print("\nPredicting values of feature '" + feature + "' in Perintal dataset...")
    pred_feature = model.predict(pred_dataset)
    if (feature not in ['fumam', 'fumap', 'alcoholm', 'alcoholp']):  # Features categóricas
        label_mapping_file = open(os.path.join(model_folder, "label_mapping_" + feature[:-1] + ".json"), "r")
        label_mapping = label_mapping_file.read()
        label_mapping = json.loads(label_mapping)
        pred_feature_codes = []
        for pred in pred_feature:
            code = list(label_mapping.keys())[list(label_mapping.values()).index(pred)]
            pred_feature_codes.append(code)
        pred_feature = pred_feature_codes
    dataset[feature] = pred_feature
    dataset['pesorec'] = pesorec
    return dataset


def histogram(dataset, feature_name):
    class_name = dataset.columns[-1]
    column = dataset[feature_name].map(str)
    sns.histplot(column[dataset[class_name] == 1], color='green', label='NORMAL weight', alpha=0.5)
    sns.histplot(column[dataset[class_name] == 0], color='red', label='LOW weight', alpha=0.5)
    sns.histplot(column[dataset[class_name] == 2], color='orange', label='HIGH weight', alpha=0.5)
    plt.xlabel(feature_name)
    plt.legend()
    plt.title('Histogram of feature: ' + feature_name)
    plt.savefig(os.path.join(output_dir, "Distribution_" + feature_name + ".png"))
    plt.tight_layout()
    print("\nHistogram of feature '" + feature_name + "' saved in: " + os.path.join(output_dir, "Distribution_" + feature_name + "_perinatal.png"))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to add to Perinatal dataset new predictions of ENSE features: 'fumam' (mother's tobacco use), 'fumap' (father's tobacco use), 'alcoholm' (mother's alcohol use), and 'alcoholp' (father's alcohol use). Usage example: $python add_ense_features_to_perinatal.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv dataPerinatal_predicted.csv -o pathTo/PerinatalWithENSE")
    parser.add_argument("input_dataset", help="Path to file with input ENSE dataset: 'dataENSE2017_converted.csv'")
    parser.add_argument("input_dataset_compatible_m",
                        help="Path to file with input women ENSE dataset compatible with Perinatal dataset: 'dataENSE2017_compatible_m.csv'")
    parser.add_argument("input_dataset_compatible_p",
                        help="Path to file with input men ENSE dataset compatible with Perinatal dataset: 'dataENSE2017_compatible_p.csv'")
    parser.add_argument("input_perinatal_dataset", help="Path to file with input Perinatal dataset. For example: 'dataPerinatal_predictions.csv'")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for created Perinatal dataset with ENSE features and predictive models. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_dataset = args['input_dataset']
    input_dataset_compatible_m = args['input_dataset_compatible_m']
    input_dataset_compatible_p = args['input_dataset_compatible_p']
    input_perinatal_dataset = args['input_perinatal_dataset']
    output_dir = args['output_dir']

    print('\nReading ENSE dataset from: ' + str(input_dataset))
    dataset_ENSE = pd.read_csv(input_dataset)

    print('\nReading ENSE dataset compatible with Perinatal dataset from: ' + str(input_dataset_compatible_m))
    dataset_compatible_m = pd.read_csv(input_dataset_compatible_m, index_col=0)

    print('\nReading ENSE dataset compatible with Perinatal dataset from: ' + str(input_dataset_compatible_p))
    dataset_compatible_p = pd.read_csv(input_dataset_compatible_p, index_col=0)


    Path(os.path.join(output_dir, "ModelsFeaturePredictors")).mkdir(parents=True, exist_ok=True)

    # FumaPredictor for WOMEN
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "fumam")).mkdir(parents=True, exist_ok=True)
    output_dir_fumam = os.path.join(output_dir, "ModelsFeaturePredictors", "fumam")
    print("\nTraining 'fumam' feature predictor model...")
    dataset_fuma_m = fumaPredictor.create_dataset(dataset_ENSE, dataset_compatible_m)
    distribution_fuma_m(dataset_fuma_m, output_dir_fumam)
    Path(os.path.join(output_dir_fumam, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_rf = os.path.join(output_dir_fumam, "RandomForest")
    rf_data = fumaPredictor.randomforest_predictor(dataset_fuma_m, output_dir_aux_rf)
    Path(os.path.join(output_dir_fumam, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_dnn = os.path.join(output_dir_fumam, "DeepNeuralNetwork")
    dnn_data = fumaPredictor.dnn_predictor(dataset_fuma_m, output_dir_aux_dnn)
    fumaPredictor.model_comparison(rf_data, dnn_data, output_dir_fumam)

    # FumaPredictor for MEN
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "fumap")).mkdir(parents=True, exist_ok=True)
    output_dir_fumap = os.path.join(output_dir, "ModelsFeaturePredictors", "fumap")
    print("\nTraining 'fumap' feature predictor model...")
    dataset_fuma_p = fumaPredictor.create_dataset(dataset_ENSE, dataset_compatible_p)
    distribution_fuma_p(dataset_fuma_p, output_dir_fumap)
    Path(os.path.join(output_dir_fumap, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_rf = os.path.join(output_dir_fumap, "RandomForest")
    rf_data = fumaPredictor.randomforest_predictor(dataset_fuma_p, output_dir_aux_rf)
    Path(os.path.join(output_dir_fumap, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_dnn = os.path.join(output_dir_fumap, "DeepNeuralNetwork")
    dnn_data = fumaPredictor.dnn_predictor(dataset_fuma_p, output_dir_aux_dnn)
    fumaPredictor.model_comparison(rf_data, dnn_data, output_dir_fumap)

    # AlcoholPredictor for WOMEN
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholm")).mkdir(parents=True, exist_ok=True)
    output_dir_alcoholm = os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholm")
    print("\nTraining 'alcoholm' feature predictor model...")
    dataset_alcohol_m = alcoholPredictor.create_dataset(dataset_ENSE, dataset_compatible_m)
    distribution_alcohol_m(dataset_alcohol_m, output_dir_alcoholm)
    Path(os.path.join(output_dir_alcoholm, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_rf = os.path.join(output_dir_alcoholm, "RandomForest")
    rf_data = alcoholPredictor.randomforest_predictor(dataset_alcohol_m, output_dir_aux_rf)
    Path(os.path.join(output_dir_alcoholm, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_dnn = os.path.join(output_dir_alcoholm, "DeepNeuralNetwork")
    dnn_data = alcoholPredictor.dnn_predictor(dataset_alcohol_m, output_dir_aux_dnn)
    alcoholPredictor.model_comparison(rf_data, dnn_data, output_dir_alcoholm)

    # AlcoholPredictor for MEN
    Path(os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholp")).mkdir(parents=True, exist_ok=True)
    output_dir_alcoholp = os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholp")
    print("\nTraining 'alcoholp' feature predictor model...")
    dataset_alcohol_p = alcoholPredictor.create_dataset(dataset_ENSE, dataset_compatible_p)
    distribution_alcohol_p(dataset_alcohol_p, output_dir_alcoholp)
    Path(os.path.join(output_dir_alcoholp, "RandomForest")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_rf = os.path.join(output_dir_alcoholp, "RandomForest")
    rf_data = alcoholPredictor.randomforest_predictor(dataset_alcohol_p, output_dir_aux_rf)
    Path(os.path.join(output_dir_alcoholp, "DeepNeuralNetwork")).mkdir(parents=True, exist_ok=True)
    output_dir_aux_dnn = os.path.join(output_dir_alcoholp, "DeepNeuralNetwork")
    dnn_data = alcoholPredictor.dnn_predictor(dataset_alcohol_p, output_dir_aux_dnn)
    alcoholPredictor.model_comparison(rf_data, dnn_data, output_dir_alcoholp)

    print('\nReading Perinatal dataset from: ' + str(input_perinatal_dataset))
    dataset_perinatal = pd.read_csv(input_perinatal_dataset, index_col=0)

    # Predecir los missing values con los modelos entrenados
    # fumam
    model_fumam = joblib.load(os.path.join(output_dir, "ModelsFeaturePredictors", "fumam", "RandomForest", "RF_MODEL_fuma.joblib"))
    dataset_predicted_perinatal = predictions(dataset_perinatal, os.path.join(output_dir, "ModelsFeaturePredictors", "fumam", "RandomForest"), model_fumam, "fumam")
    histogram(dataset_predicted_perinatal, 'fumam')

    # fumap
    model_fumap = joblib.load(os.path.join(output_dir, "ModelsFeaturePredictors", "fumap", "RandomForest", "RF_MODEL_fuma.joblib"))
    dataset_predicted_perinatal = predictions(dataset_predicted_perinatal, os.path.join(output_dir, "ModelsFeaturePredictors", "fumap", "RandomForest"), model_fumap, "fumap")
    histogram(dataset_predicted_perinatal, 'fumap')

    # alcoholm
    model_alcoholm = joblib.load(os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholm", "RandomForest", "RF_MODEL_alcohol.joblib"))
    dataset_predicted_perinatal = predictions(dataset_predicted_perinatal, os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholm", "RandomForest"), model_alcoholm, "alcoholm")
    histogram(dataset_predicted_perinatal, 'alcoholm')

    # alcoholp
    model_alcoholp = joblib.load(os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholp", "RandomForest", "RF_MODEL_alcohol.joblib"))
    dataset_predicted_perinatal = predictions(dataset_predicted_perinatal, os.path.join(output_dir, "ModelsFeaturePredictors", "alcoholp", "RandomForest"), model_alcoholp, "alcoholp")
    histogram(dataset_predicted_perinatal, 'alcoholp')

    dataset_name = Path(input_perinatal_dataset).stem
    dataset_predicted_perinatal.to_csv(os.path.join(output_dir, dataset_name + "_ENSE.csv"))
    print("Dataset with predictions of features 'fumam', 'fumap', 'alcoholm' and 'alcoholp' saved in: " + os.path.join(output_dir, dataset_name + "_ENSE.csv"))