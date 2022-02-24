import argparse
import os
import pandas as pd
import warnings

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def create_datasets_fuma_m(dataset_ENSE, dataset_compatible_m):
    dataset_ENSE_result = dataset_ENSE.copy()
    dataset_compatible_m_result = dataset_compatible_m.copy()

    # Dataset de MUJERES con las features de ENSE
    dataset_ENSE_result = dataset_ENSE_result.loc[dataset_ENSE_result['SEXOa'] == 2]
    v121_values = dataset_ENSE_result["V121"]
    dataset_ENSE_result['fuma'] = v121_values
    dataset_ENSE_result = dataset_ENSE_result[~dataset_ENSE_result['fuma'].isin([8,9])]
    dataset_ENSE_result['fuma'] = dataset_ENSE_result['fuma'].replace([3, 4], 0)
    dataset_ENSE_result['fuma'] = dataset_ENSE_result['fuma'].replace([1, 2], 1)
    dataset_ENSE_result = dataset_ENSE_result.drop(["V121", "SEXOa"], 1)


    # Dataset de MUJERES con las features de Perinatal
    indices = dataset_compatible_m_result.index.values.tolist()
    v121_values = dataset_ENSE["V121"].loc[indices]
    dataset_compatible_m_result['fuma'] = v121_values
    dataset_compatible_m_result = dataset_compatible_m_result.dropna()
    dataset_compatible_m_result = dataset_compatible_m_result[~dataset_compatible_m_result['fuma'].isin([8,9])]
    dataset_compatible_m_result['fuma'] = dataset_compatible_m_result['fuma'].replace([3,4], 0)
    dataset_compatible_m_result['fuma'] = dataset_compatible_m_result['fuma'].replace([1, 2], 1)

    return dataset_ENSE_result, dataset_compatible_m_result


def create_datasets_fuma_p(dataset_ENSE, dataset_compatible_p):
    dataset_ENSE_result = dataset_ENSE.copy()
    dataset_compatible_p_result = dataset_compatible_p.copy()

    # Dataset de HOMBRES con las features de ENSE
    dataset_ENSE_result = dataset_ENSE_result.loc[dataset_ENSE_result['SEXOa'] == 1]
    v121_values = dataset_ENSE_result["V121"]
    dataset_ENSE_result['fuma'] = v121_values
    dataset_ENSE_result = dataset_ENSE_result[~dataset_ENSE_result['fuma'].isin([8,9])]
    dataset_ENSE_result['fuma'] = dataset_ENSE_result['fuma'].replace([3, 4], 0)
    dataset_ENSE_result['fuma'] = dataset_ENSE_result['fuma'].replace([1, 2], 1)
    dataset_ENSE_result = dataset_ENSE_result.drop(["V121", "SEXOa"], 1)


    # Dataset de HOMBRES con las features de Perinatal
    indices = dataset_compatible_p_result.index.values.tolist()
    v121_values = dataset_ENSE["V121"].loc[indices]
    dataset_compatible_p_result['fuma'] = v121_values
    dataset_compatible_p_result = dataset_compatible_p_result.dropna()
    dataset_compatible_p_result = dataset_compatible_p_result[~dataset_compatible_p_result['fuma'].isin([8,9])]
    dataset_compatible_p_result['fuma'] = dataset_compatible_p_result['fuma'].replace([3,4], 0)
    dataset_compatible_p_result['fuma'] = dataset_compatible_p_result['fuma'].replace([1, 2], 1)

    return dataset_ENSE_result, dataset_compatible_p_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to create the ENSE datasets with 'fuma' class for women (mothers) and men (fathers). Two datasets are created for each sex: one containing all features of ENSE dataset, and other only with features compatibles with Perinatal dataset. Usage example: $python ense_fuma_datasets_creation.py dataENSE2017_converted.csv dataENSE2017_compatible_m.csv dataENSE2017_compatible_p.csv -o pathTo/FumaDatasets")
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

    print("\nCreating ENSE datasets with 'fuma' class for women...")
    dataset_ENSE_m_fuma, dataset_compatible_m_fuma = create_datasets_fuma_m(dataset_ENSE, dataset_compatible_m)
    print("Saving ENSE women dataset in: " + str(os.path.join(output_dir, "dataENSE2017_m_fuma.csv")))
    dataset_ENSE_m_fuma.to_csv(os.path.join(output_dir, "dataENSE2017_m_fuma.csv"))
    print("Saving ENSE women dataset compatible with Perinatal in: " + str(os.path.join(output_dir, "dataENSE2017_compatible_m_fuma.csv")))
    dataset_compatible_m_fuma.to_csv(os.path.join(output_dir, "dataENSE2017_compatible_m_fuma.csv"))

    print("\nCreating ENSE datasets with 'fuma' class for men...")
    dataset_ENSE_p_fuma, dataset_compatible_p_fuma = create_datasets_fuma_p(dataset_ENSE, dataset_compatible_p)
    print("Saving ENSE men dataset in: " + str(os.path.join(output_dir, "dataENSE2017_p_fuma.csv")))
    dataset_ENSE_p_fuma.to_csv(os.path.join(output_dir, "dataENSE2017_p_fuma.csv"))
    print("Saving ENSE men dataset compatible with Perinatal in: " + str(os.path.join(output_dir, "dataENSE2017_compatible_p_fuma.csv")))
    dataset_compatible_p_fuma.to_csv(os.path.join(output_dir, "dataENSE2017_compatible_p_fuma.csv"))