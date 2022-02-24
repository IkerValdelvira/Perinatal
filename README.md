# PERINATAL

Software developed to carry out ***PERINATAL project***. Here you will find all the scripts developed during this research work with the following objective: creation of birth weight classifier (LOW, NORMAL or HIGH weight) using predictor variables prior to childbirth, and mother and father's socioeconomic features. The predictive model is trained with the data of all births in Spain between the years 1996 and 2019.

Weight classification is done as follows (class *'pesorec'*) :
* LOW weight: <2500g
* NORMAL weight: \[2500g, 4000g\]
* HIGH weight: >4000g

Further predictive models have been trained adding some predicted features from the ENSE 2017 (National Health Survey of Spain in 2017): mother's tobacco use, father's tobacco use, mother's alcohol use, and father's alcohol use.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Software description and usage](#software-description-and-usage)
<br />2.1. [*PreprocessAndExperiments* package scripts](#preprocessandexperiments-package-scripts)
<br />2.2. [*TrainAndPredict* package scripts](#trainandpredict-package-scripts)
3. [TUTORIAL: Making predictions using trained models](#tutorial-making-predictions-using-trained-models)
4. [PROTOTYPES: Trained and saved models](#prototypes-trained-and-saved-models)
5. [Project documentation](#project-documentation)
6. [Author and contact](#author-and-contact)

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
In this package, scripts for the data preprocessing and scripts to execute all the experiments carried out are saved.

For example, the following two scripts are needed to create the Perinatal datasets pre-processed and ready to train predictive models:

* ***PreprocessAndExperiments/dataset_creation.py***: Script to create the Perinatal dataset. Usage example: *$python PreprocessAndExperiments/dataset_creation.py dataPerinatal.csv -o pathTo/Perinatal*

* ***PreprocessAndExperiments/dataset_preprocessing.py***: Script to preprocess Perinatal dataset. 3 dataset will be created: 'dataPerinatal_remove_items.csv' removing all items with missing values, 'dataPerinatal_remove_features.csv' removing features with missing values, and 'dataPerinatal_predicted.csv' predicting missing values. Usage example: *$python dataset_preprocessing.py dataPerinatal_converted.csv -o pathTo/Preprocess*


To create ENSE 2017 dataset and sets with classes *'fuma'* / *'alcohol'* compatible with Perinatal dataset, following scripts are needed:

* ***PreprocessAndExperiments/ENSE/ense_dataset_creation.py***: Script to create the ENSE dataset, and datasets with features compatibles with Perinatal dataset. Usage example: *$python ense_dataset_creation.py dataENSE2017.txt -o pathTo/ENSE*

* ***PreprocessAndExperiments/ENSE/ense_fuma_datasets_creation.py***: Script to create the ENSE datasets with *'fuma'* class for women (mothers) and men (fathers). Two datasets are created for each sex: one containing all features of ENSE dataset, and other only with features compatibles with Perinatal dataset. Usage example: *$python ense_fuma_datasets_creation.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv -o pathTo/FumaDatasets*

* ***PreprocessAndExperiments/ENSE/ense_alcohol_datasets_creation.py***: Script to create the ENSE datasets with *'alcohol'* class for women (mothers) and men (fathers). Two datasets are created for each sex: one containing all features of ENSE dataset, and other only with features compatibles with Perinatal dataset. Usage example: *$python ense_alcohol_datasets_creation.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv -o pathTo/AlcoholDatasets*


Finally, to add predictions of features *'fumam'* / *'fumap'* / *'alcoholm'* / *'alcoholp'* from the ENSE 2017 to the Perinatal data set, it is necessary to execute the following script:

* ***PreprocessAndExperiments/ENSE/add_ense_features_to_perinatal.py***: Script to add to Perinatal dataset new predictions of ENSE features: *'fumam'* (mother's tobacco use), *'fumap'* (father's tobacco use), *'alcoholm'* (mother's alcohol use), and *'alcoholp'* (father's alcohol use). Usage example: *$python add_ense_features_to_perinatal.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv dataPerinatal_predicted.csv -o pathTo/PerinatalWithENSE*


The rest of scripts in *PreprocessAndExperiments* package belong to experiments carried out for the creation of birth weight predictive models and predictions in some ENSE 2017 features.


### *TrainAndPredict* package scripts:

The two scripts to train a birth weight predictive model with a specific dataset and make predictions in new items are saved in this package.

* ***TrainAndPredict/train_model_pesorec.py***: Script to train a *'pesorec'* classification model with the entered dataset, and using RandomForest and/or optimized DNN. Resampling techniques can be applied like oversampling, undersampling and over/undersampling. Usage example: *$python train_model_pesorec.py dataPerinatal.csv -a DNN -o pathTo/ModelFolder*

* ***TrainAndPredict/make_predictions_pesorec.py***: Script to make predictions on new input items using a *'pesorec'* predictive model. Usage example: *$python make_predictions_pesorec.py new_items.csv pathTo/ModelFolder -o pathTo/Predictions*


## TUTORIAL: Making predictions using trained models

This section explains how to make birth weight predictions on new data using one of the trained *'pesorec'* (birth weight) models.

**1. SELECT ONE OF THE TRAINED BIRTH WEIGHT MODELS**

Trained models are located in the [***models***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/) directory. For example, we are going to use the model with the socioeconomic features of Perinatal dataset and added features related to tobacco and alcohol use predicted from the ENSE 2017 dataset: [***'pesorec' model with ENSE features***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec_ENSE/Model)


**2. CREATE CSV FILE CONTAINING NEW ITEMS TO PREDICT**

We have to create a file in CSV format with the instances we want to classify. Inside the folder of each model, there is an example CSV to insert the new instances to predict, which can be used as a template. For the [***'pesorec' model with ENSE features***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec_ENSE/Model) example, we have [this template](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec_ENSE/new_items_withENSE.csv) .

Following image shows the structure of the CSV file with some new items to be predicted:

![alt text](https://github.com/IkerValdelvira/Perinatal/blob/master/example_images/new_items_example.png?raw=true)

**3. MAKE PREDICTIONS ON NEW ITEMS**

Once we have the predictive model and the corresponding CSV file with new items, the following script must be executed to make the predictions:
```
$ python TrainAndPredict/make_predictions_pesorec.py PATH_TO/new_items.csv PATH_TO/ModelFolder -o OUTPUT_DIR
```

A file called ***PREDICTIONS.txt*** will be created, which contains predictions made by the model on new input items. In this file appears the ID (index) of each new item introduced together with the outcome of the model (prediction). Following image shows the output *PREDICTIONS.txt* file after having introduced [***new_items_withENSE.csv***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec_ENSE/new_items_withENSE.csv):

![alt text](https://github.com/IkerValdelvira/Perinatal/blob/master/example_images/predictions_example.png?raw=true)


## PROTOTYPES: Trained and saved models

Prototypes of trained predictive models are saved in [***models***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/) folder. Each model has a CSV template to introduce new items to be predicted, for example: [***new_items.csv***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec/Models_OriginalDistribution/new_items.csv).

* [***Models_Pesorec***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec): In this folder, all *'pesorec'* (birth weight) predictive models trained with the original Perinatal dataset are saved. Different models depend on the data distribution used for their training: original distribution, oversampling, undersampling, oversampling (10%) / undersampling (50%), and multiple custom undersampling.
* [***Models_Pesorec_PesonSemanas***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec_PesonSemanas): A *'pesorec'* (birth weight) predictive model adding *peson_semanas* feature is saved in this folder. This *fake feature* represents the weight gain of the fetus per week, assuming weight and time of pregnancy were linear.
* [***Models_Pesorec_PesoSemanaXX***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec_PesoSemanaXX): *peso_semana_XX* feature has been added to *'pesorec'* (birth weight) predictive models in this folder, which calculates an estimate of fetus weight in week XX of pregnancy. Two types of models have been trained: a baseline only with the age of the mother, and another model adding Perinatal socioeconomic features.
* [***Models_Pesorec_ENSE***](https://github.com/IkerValdelvira/Perinatal/blob/master/models/Models_Pesorec_ENSE): In this folder, *'pesorec'* (birth weight) predictive model trained with original features of Perintal dataset and adding predicted features from the ENSE 2017 dataset is saved. Added features are indicators of tobacco and alcohol use by newborn's mother and father.


## Project documentation:

The documentation describing the work in this project can be found here: [Documentacion_Perinatal_IkerValdelvira.pdf](https://github.com/IkerValdelvira/Perinatal/blob/master/documentation/Documentacion_Perinatal_IkerValdelvira.pdf)


## Author and contact:

Iker Valdelvira ([ivaldelvira001@ikasle.ehu.eus](mailto:ivaldelvira001@ikasle.ehu.eus))
