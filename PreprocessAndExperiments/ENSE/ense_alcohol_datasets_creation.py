import argparse
import os
import pandas as pd
import warnings

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def create_datasets_alcohol_m(dataset_ENSE, dataset_compatible_m):
    dataset_ENSE_result = dataset_ENSE.copy()
    dataset_compatible_m_result = dataset_compatible_m.copy()

    # Dataset de MUJERES con las features de ENSE
    dataset_ENSE_result = dataset_ENSE_result.loc[dataset_ENSE_result['SEXOa'] == 2]
    w127_values = dataset_ENSE_result["W127"]
    dataset_ENSE_result['alcohol'] = w127_values
    dataset_ENSE_result = dataset_ENSE_result[~dataset_ENSE_result['alcohol'].isin([98,99])]
    dataset_ENSE_result['alcohol'] = dataset_ENSE_result['alcohol'].replace([5, 6, 7, 8, 9], 0)  # Menos de 1-2 días a la semana
    dataset_ENSE_result['alcohol'] = dataset_ENSE_result['alcohol'].replace([1, 2, 3, 4], 1)  # Mínimo 1-2 días a la semana
    dataset_ENSE_result = dataset_ENSE_result.drop(["W127", "SEXOa"], 1)


    # Dataset de MUJERES con las features de Perinatal
    indices = dataset_compatible_m_result.index.values.tolist()
    w127_values = dataset_ENSE["W127"].loc[indices]
    dataset_compatible_m_result['alcohol'] = w127_values
    dataset_compatible_m_result = dataset_compatible_m_result.dropna()
    dataset_compatible_m_result = dataset_compatible_m_result[~dataset_compatible_m_result['alcohol'].isin([98, 99])]
    dataset_compatible_m_result['alcohol'] = dataset_compatible_m_result['alcohol'].replace([5, 6, 7, 8, 9], 0)  # Menos de 1-2 días a la semana
    dataset_compatible_m_result['alcohol'] = dataset_compatible_m_result['alcohol'].replace([1, 2, 3, 4], 1)  # Mínimo 1-2 días a la semana

    return dataset_ENSE_result, dataset_compatible_m_result


def create_datasets_alcohol_p(dataset_ENSE, dataset_compatible_p):
    dataset_ENSE_result = dataset_ENSE.copy()
    dataset_compatible_p_result = dataset_compatible_p.copy()

    # Dataset de HOMBRES con las features de ENSE
    dataset_ENSE_result = dataset_ENSE_result.loc[dataset_ENSE_result['SEXOa'] == 1]
    w127_values = dataset_ENSE_result["W127"]
    dataset_ENSE_result['alcohol'] = w127_values
    dataset_ENSE_result = dataset_ENSE_result[~dataset_ENSE_result['alcohol'].isin([8,9])]
    dataset_ENSE_result['alcohol'] = dataset_ENSE_result['alcohol'].replace([5, 6, 7, 8, 9], 0)  # Menos de 1-2 días a la semana
    dataset_ENSE_result['alcohol'] = dataset_ENSE_result['alcohol'].replace([1, 2, 3, 4], 1)  # Mínimo 1-2 días a la semana
    dataset_ENSE_result = dataset_ENSE_result.drop(["W127", "SEXOa"], 1)


    # Dataset de HOMBRES con las features de Perinatal
    indices = dataset_compatible_p_result.index.values.tolist()
    w127_values = dataset_ENSE["W127"].loc[indices]
    dataset_compatible_p_result['alcohol'] = w127_values
    dataset_compatible_p_result = dataset_compatible_p_result.dropna()
    dataset_compatible_p_result = dataset_compatible_p_result[~dataset_compatible_p_result['alcohol'].isin([98, 99])]
    dataset_compatible_p_result['alcohol'] = dataset_compatible_p_result['alcohol'].replace([5, 6, 7, 8, 9], 0)  # Menos de 1-2 días a la semana
    dataset_compatible_p_result['alcohol'] = dataset_compatible_p_result['alcohol'].replace([1, 2, 3, 4], 1)  # Mínimo 1-2 días a la semana

    return dataset_ENSE_result, dataset_compatible_p_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to create the ENSE datasets with 'alcohol' class for women (mothers) and men (fathers). Two datasets are created for each sex: one containing all features of ENSE dataset, and other only with features compatibles with Perinatal dataset. Usage example: $python ense_alcohol_datasets_creation.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv -o pathTo/AlcoholDatasets")
    parser.add_argument("input_dataset", help="Path to file with input ENSE dataset: 'dataENSE2017_converted.csv'")
    parser.add_argument("input_dataset_compatible_m",
                        help="Path to file with input women ENSE dataset compatible with Perinatal dataset: 'dataENSE2017_compatible_m.csv'")
    parser.add_argument("input_dataset_compatible_p",
                        help="Path to file with input men ENSE dataset compatible with Perinatal dataset: 'dataENSE2017_compatible_p.csv'")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for created datasets. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_dataset = args['input_dataset']
    input_dataset_compatible_m = args['input_dataset_compatible_m']
    input_dataset_compatible_p = args['input_dataset_compatible_p']
    output_dir = args['output_dir']

    print('\nReading ENSE dataset from: ' + str(input_dataset))
    dataset_ENSE = pd.read_csv(input_dataset)

    print('\nReading women ENSE dataset compatible with Perinatal dataset from: ' + str(input_dataset_compatible_m))
    dataset_compatible_m = pd.read_csv(input_dataset_compatible_m, index_col=0)

    print('\nReading men ENSE dataset compatible with Perinatal dataset from: ' + str(input_dataset_compatible_p))
    dataset_compatible_p = pd.read_csv(input_dataset_compatible_p, index_col=0)

    print("\nCreating ENSE datasets with 'alcohol' class for women...")
    dataset_ENSE_m_alcohol, dataset_compatible_m_alcohol = create_datasets_alcohol_m(dataset_ENSE, dataset_compatible_m)
    print("Saving ENSE women dataset in: " + str(os.path.join(output_dir, "dataENSE2017_m_alcohol.csv")))
    dataset_ENSE_m_alcohol.to_csv(os.path.join(output_dir, "dataENSE2017_m_alcohol.csv"))
    print("Saving ENSE women dataset compatible with Perinatal in: " + str(os.path.join(output_dir, "dataENSE2017_compatible_m_alcohol.csv")))
    dataset_compatible_m_alcohol.to_csv(os.path.join(output_dir, "dataENSE2017_compatible_m_alcohol.csv"))

    print("\nCreating ENSE datasets with 'alcohol' class for men...")
    dataset_ENSE_p_alcohol, dataset_compatible_p_alcohol = create_datasets_alcohol_p(dataset_ENSE, dataset_compatible_p)
    print("Saving ENSE men dataset in: " + str(os.path.join(output_dir, "dataENSE2017_p_alcohol.csv")))
    dataset_ENSE_p_alcohol.to_csv(os.path.join(output_dir, "dataENSE2017_p_alcohol.csv"))
    print("Saving ENSE men dataset compatible with Perinatal in: " + str(os.path.join(output_dir, "dataENSE2017_compatible_p_alcohol.csv")))
    dataset_compatible_p_alcohol.to_csv(os.path.join(output_dir, "dataENSE2017_compatible_p_alcohol.csv"))