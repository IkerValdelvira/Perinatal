# PERINATAL

Software developed to carry out ***PERINATAL project***. Here you will find all the scripts developed during this research work with the following objective: the creation of birth weight classifier (LOW, NORMAL or HIGH weight) using predictor variables prior to childbirth and socioeconomic variables of the mother and father. The predictive model is trained with the data of all births in Spain between the years 1996 and 2019.

Weight classification is done as follows:
* LOW weight: <2500g
* NORMAL weight: \[2500g, 4000g\]
* HIGH weight: >4000g

Further predictive models have been trained adding some predicted variables from the ENSE 2017 (National Health Survey of Spain in 2017): mother's tobacco use, father's tobacco use, mother's alcohol use, and father's alcohol use.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Software description and usage](#software-description-and-usage)
<br />2.1. [*PreprocessAndExperiments* package scripts](#preprocessandexperiments-package-scripts)
<br />2.2. [*TrainAndPredict* package scripts](#trainandpredict-package-scripts)
3. [TUTORIAL: Getting PRAFAI predictive model and making predictions on new items](#tutorial-getting-prafai-predictive-model-and-making-predictions-on-new-items)
4. [Project documentation](#project-documentation)
5. [Author and contact](#author-and-contact)

## Prerequisites:

It is recommended to use the application in an environment with **Python 3.8** and the installation of the following packages is required:

* imbalanced_learn==0.8.1
* imblearn==0.0
* joblib==1.1.0
* keras==2.7.0
* keras_tuner==1.1.0
* matplotlib==3.4.3
* numpy==1.21.3
* pandas==1.3.4
* scikit_learn==1.0.2
* seaborn==0.11.2
* tabulate==0.8.9
* tensorflow==2.7.0
* tensorflow_addons==0.15.0
* openpyxl==3.0.9

To install the necessary packages with the corresponding versions automatically, execute the following command:

```
$ pip install -r requirements.txt
```

## Software description and usage:

There is a help option that shows the usage of each script in the application, e.g. in *dataset_creation.py* script:

```
$ python PreprocessAndExperiments/dataset_creation.py -h
```
  
### *PreprocessAndExperiments* package scripts:
In this package are scripts for the data preprocessing and scripts to execute all the experiments carried out.

For example, the following two scripts are needed to create the Perinatal datasets pre-processed and ready to train the predictive models:

* ***PreprocessAndExperiments/dataset_creation.py***: Script to create the Perinatal dataset. Usage example: *$python PreprocessAndExperiments/dataset_creation.py dataPerinatal.csv -o pathTo/Perinatal*

* ***PreprocessAndExperiments/dataset_preprocessing.py***: Script to preprocess Perinatal dataset. 3 dataset will be created: 'dataPerinatal_remove_items.csv' removing all items with missing values, 'dataPerinatal_remove_features.csv' removing features with missing values, and 'dataPerinatal_predicted.csv' predicting missing values. Usage example: *$python dataset_preprocessing.py dataPerinatal_converted.csv -o pathTo/Preprocess*


To create ENSE 2017 dataset and sets with classes *'fuma'* / *'alcohol'* compatible with Perinatal dataset following scripts are needed:

* ***PreprocessAndExperiments/ENSE/ense_dataset_creation.py***: Script to create the ENSE dataset, and datasets with features compatibles with Perinatal dataset. Usage example: *$python ense_dataset_creation.py dataENSE2017.txt -o pathTo/ENSE*

* ***PreprocessAndExperiments/ENSE/ense_fuma_datasets_creation.py***: Script to create the ENSE datasets with *'fuma'* class for women (mothers) and men (fathers). Two datasets are created for each sex: one containing all features of ENSE dataset, and other only with features compatibles with Perinatal dataset. Usage example: *$python ense_fuma_datasets_creation.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv -o pathTo/FumaDatasets*

* ***PreprocessAndExperiments/ENSE/ense_alcohol_datasets_creation.py***: Script to create the ENSE datasets with *'alcohol'* class for women (mothers) and men (fathers). Two datasets are created for each sex: one containing all features of ENSE dataset, and other only with features compatibles with Perinatal dataset. Usage example: *$python ense_alcohol_datasets_creation.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv -o pathTo/AlcoholDatasets*


Finally, to add the variables *'fumam'* / *'fumap'* / *'alcoholm'* / *'alcoholp'* from the ENSE 2017 to the Perinatal data set, it is necessary to execute the following script:

* ***PreprocessAndExperiments/ENSE/add_ense_features_to_perinatal.py***: Script to add to Perinatal dataset new predictions of ENSE features: *'fumam'* (mother's tobacco use), *'fumap'* (father's tobacco use), *'alcoholm'* (mother's alcohol use), and *'alcoholp'* (father's alcohol use). Usage example: *$python add_ense_features_to_perinatal.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv dataPerinatal_predicted.csv -o pathTo/PerinatalWithENSE*


The rest of scripts in *PreprocessAndExperiments* package belong to experiments carried out for the creation of predictive models of birth weight and some variables of the ENSE 2017.


### *TrainAndPredict* package scripts:

* ***musexmlex.py***: Script to extract an 12-lead ECG rhythm strip from a MUSE(R) XML file. It converts MUSE-XML files to CSV files. Credits to [***PROJECT: musexmlexport***](https://github.com/rickead/musexmlexport).

* ***train_models.py***: Script to train and evaluate different AF classification models based on 12-lead ECGs: XGBoost, FCN, FCN+MLP(age,sex), Encoder, Encoder+MLP(age,sex), FCN+Encoder, FCN+Encoder+MLP(age,sex) or LSTM.

* ***train_FCN_MLP_CV.py***: Script to train FCN+MLP(age,sex) AF classification model based on 12-lead ECGs and evaluate it via 10-fold Cross Validation.

* ***reductionFCN_MLP.py***: Script in which the AF classification model development experiment is performed by reducing the number of training ECGs. FCN+MLP(age,sex) classification algorithm is used.


## TUTORIAL: Getting PRAFAI predictive model and making predictions on new items:

This section explains how to obtain the final predictive model and make predictions on new data.

**1. CREATE PRAFAI DATASET**

```
$ python PRAFAI/dataset_creation.py INPUT_DIR -o OUTPUT_DIR
```

**2. TRAIN AND GET PRAFAI PREDICTIVE MODEL**

```
$ python PRAFAI/best_model.py PATH_TO/dataset.csv -o OUTPUT_DIR
```
A folder called ***FinalModel*** will be created, among others, which contains the model and necessary files to make new predictions.


**3. MAKE PREDICTIONS ON NEW ITEMS**

To make a prediction on a new item, the PRAFAI model needs the values of the following 30 features:

* Numeric features:<br />**'potasio'**, **'no_hdl'**, **'colesterol'**, **'ntprobnp'**, **'vsg'**, **'fevi'**, **'diametro_ai'**, **'area_ai'**, **'numero_dias_desde_ingreso_hasta_evento'**, **'numero_dias_ingresado'** and **'edad'**.

* Binary features:<br />**'ablacion'**, **'ansiedad'**, **'demencia'**, **'sahos'**, **'hipertiroidismo'**, **'cardiopatia_isquemica'**, **'valvula_mitral_reumaticas'**, **'genero'**, **'pensionista'**, **'residenciado'**, **'n05a'**, **'n05b'**, **'c01'**, **'c01b'**, **'c02'**, **'c04'**, **'c09'**, **'c10'** and **'polimedicacion'**.

\* In **'genero'** (genre) feature **female** is set as **0** and **male** is set as **1**.
<br />\* The new items that are introduced can have missing values in any of the features. These missing values will be automatically handled to make the predictions, but the fewer missing values there are, the more reliable the predictions will be.

The new items to be predicted must be introduced in a **CSV file** delimited by comma (**,**). The first column (index) must be the ID of the new item. A template and example of this CSV file is available at: [new_items_template.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_template.csv) and [new_items_example.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_example.csv).

Following image shows the structure of a CSV file with some new items to be predicted ([new_items_example.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_example.csv)):

![alt text](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/example_images/new_items_example.png?raw=true)

To make the predictions, the following script must be executed:
```
$ python PRAFAI/make_predictions.py PATH_TO/new_items_example.csv PATH_TO/FinalModel -o OUTPUT_DIR
```

A file called ***PREDICTIONS.txt*** will be created, which contains the predictions made by the model on new input items. In this file appears the ID (index) of each new item introduced together with the outcome of the model (prediction) and its probability. Following image shows the output *PREDICTIONS.txt* file after having introduced [new_items_example.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_example.csv):

![alt text](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/example_images/predictions_example.png?raw=true)


## Project documentation:

The documentation describing the work in this project can be found here: [TFG_PRAFAI_Memoria_IkerValdelvira.pdf](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/documentation/TFG_PRAFAI_Memoria_IkerValdelvira.pdf)


## Author and contact:

Iker Valdelvira ([ivaldelvira001@ikasle.ehu.eus](mailto:ivaldelvira001@ikasle.ehu.eus))
