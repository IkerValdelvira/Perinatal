import argparse
import os

import pandas as pd
import numpy as np
from operator import itemgetter


def try_int_float(v):
    try:
        float_v = float(v)
        if(float_v.is_integer()):
            return int(float_v)
        else:
            return float_v
    except Exception:
        return None

def create_dataset(file_path):
    dataset_matrix = []
    with open(file_path) as file:
        for line in file:
            row = []
            split = list(line)
            row.append(''.join(split[0:2]))
            row.append(''.join(split[2:10]))
            row.append(''.join(split[10:12]))
            row.append(split[12])
            row.append(''.join(split[13:16]))
            row.append(split[16])
            row.append(split[17])
            row.append(split[18])
            row.append(split[19])
            row.append(''.join(split[20:22]))
            row.append(split[22])
            row.append(''.join(split[23:26]))
            row.append(split[26])
            row.append(split[27])
            row.append(split[28])
            row.append(split[29])
            row.append(split[30])
            row.append(split[31])
            row.append(''.join(split[32:34]))
            row.append(split[34])
            row.append(split[35])
            row.append(''.join(split[36:38]))
            row.append(split[38])
            row.append(split[39])
            row.append(''.join(split[40:43]))
            row.append(''.join(split[43:46]))
            row.append(split[46])
            row.append(split[47])
            row.append(split[48])
            row.append(''.join(split[49:51]))
            row.append(split[51])
            row.append(split[52])
            row.append(split[53])
            row.append(split[54])
            row.append(''.join(split[55:57]))
            row.append(''.join(split[57:60]))
            row.append(''.join(split[60:63]))
            row.append(''.join(split[63:66]))
            row.append(''.join(split[66:69]))
            row.append(split[69])
            row = row + list(itemgetter(*[*range(70,175)])(split))
            row.append(''.join(split[175:177]))
            row.append(split[177])
            row.append(''.join(split[178:180]))
            row = row + list(itemgetter(*[*range(180,221)])(split))
            row.append(''.join(split[221:223]))
            row.append(split[223])
            row.append(''.join(split[224:226]))
            row.append(split[226])
            row.append(''.join(split[227:229]))
            row.append(split[229])
            row.append(''.join(split[230:232]))
            row.append(''.join(split[232:234]))
            row.append(''.join(split[234:236]))
            row.append(''.join(split[236:238]))
            row.append(''.join(split[238:240]))
            row.append(''.join(split[240:242]))
            row = row + list(itemgetter(*[*range(242, 261)])(split))
            row.append(''.join(split[261:263]))
            row = row + list(itemgetter(*[*range(263, 283)])(split))
            row.append(''.join(split[283:285]))
            row.append(split[285])
            row.append(''.join(split[286:289]))
            row.append(split[289])
            row.append(split[290])
            row.append(''.join(split[291:293]))
            row.append(split[293])
            row.append(split[294])
            row.append(''.join(split[295:298]))
            row.append(split[298])
            row.append(split[299])
            row.append(''.join(split[300:303]))
            row.append(split[303])
            row.append(split[304])
            row.append(split[305])
            row.append(''.join(split[306:308]))
            row.append(''.join(split[308:310]))
            row.append(''.join(split[310:312]))
            row.append(''.join(split[312:314]))
            row.append(''.join(split[314:316]))
            row = row + list(itemgetter(*[*range(316, 398)])(split))
            row.append(''.join(split[398:401]))
            row.append(''.join(split[401:404]))
            row.append(split[404])
            row.append(split[405])
            row.append(split[406])
            row.append(''.join(split[407:409]))
            row.append(''.join(split[409:411]))
            row.append(split[411])
            row.append(''.join(split[412:414]))
            row.append(''.join(split[414:416]))
            row.append(split[416])
            row.append(''.join(split[417:419]))
            row.append(''.join(split[419:421]))
            row.append(''.join(split[421:423]))
            row.append(''.join(split[423:425]))
            row.append(split[425])
            row.append(''.join(split[426:428]))
            row = row + list(itemgetter(*[*range(428, 434)])(split))
            row.append(''.join(split[434:436]))
            row = row + list(itemgetter(*[*range(436, 444)])(split))
            row.append(''.join(split[444:446]))
            row.append(split[446])
            row.append(''.join(split[447:449]))
            row.append(split[449])
            row.append(split[450])
            row.append(split[451])
            row.append(''.join(split[452:454]))
            row.append(''.join(split[454:456]))
            row.append(split[456])
            row.append(split[457])
            row.append(''.join(split[458:460]))
            row.append(split[460])
            row.append(''.join(split[461:463]))
            row.append(''.join(split[463:465]))
            row.append(''.join(split[465:467]))
            row.append(''.join(split[467:469]))
            row.append(''.join(split[469:471]))
            row.append(''.join(split[471:473]))
            row.append(''.join(split[473:475]))
            row.append(split[475])
            row.append(''.join(split[476:478]))
            row.append(''.join(split[478:480]))
            row.append(''.join(split[480:482]))
            row.append(''.join(split[482:484]))
            row.append(''.join(split[484:486]))
            row.append(''.join(split[486:488]))
            row.append(''.join(split[488:490]))
            row.append(split[490])
            row.append(''.join(split[491:493]))
            row.append(''.join(split[493:495]))
            row.append(''.join(split[495:497]))
            row.append(''.join(split[497:499]))
            row.append(''.join(split[499:501]))
            row.append(''.join(split[501:503]))
            row.append(''.join(split[503:505]))
            row.append(split[505])
            row.append(''.join(split[506:508]))
            row.append(''.join(split[508:510]))
            row.append(''.join(split[510:512]))
            row.append(''.join(split[512:514]))
            row.append(''.join(split[514:516]))
            row.append(''.join(split[516:518]))
            row.append(''.join(split[518:520]))
            row.append(split[520])
            row.append(''.join(split[521:523]))
            row.append(''.join(split[523:525]))
            row.append(''.join(split[525:527]))
            row.append(''.join(split[527:529]))
            row.append(''.join(split[529:531]))
            row.append(''.join(split[531:533]))
            row.append(''.join(split[533:535]))
            row.append(split[535])
            row.append(''.join(split[536:538]))
            row.append(''.join(split[538:540]))
            row.append(''.join(split[540:542]))
            row.append(''.join(split[542:544]))
            row.append(''.join(split[544:546]))
            row.append(''.join(split[546:548]))
            row.append(''.join(split[548:550]))
            row.append(''.join(split[550:552]))
            row = row + list(itemgetter(*[*range(552, 566)])(split))
            row.append(''.join(split[566:577]))
            row.append(split[577])
            row.append(split[578])
            row.append(''.join(split[579:584]))
            row.append(''.join(split[584:589]))
            row.append(''.join(split[589:594]))
            row = [try_int_float(val) for val in row]
            dataset_matrix.append(row)

    dataset = pd.DataFrame(dataset_matrix)
    dataset.columns = ['CCAA', 'IDENTHOGAR', 'A7_2a', 'SEXOa', 'EDADa', 'ACTIVa', 'PROXY_0', 'PROXY_1', 'PROXY_2', 'PROXY_2b', 'PROXY_3b', 'PROXY_4', 'PROXY_5', 'E1_1', 'E2_1a', 'E2_1b', 'E2_1c', 'E2_1d', 'E3', 'E4', 'E4b', 'NIVEST', 'F6', 'F7', 'F8_2', 'F9_2', 'F10', 'F11', 'F12', 'F13', 'F14a', 'F14b', 'F15', 'F16', 'F17', 'F18a_2', 'F18b_2', 'F19a_2', 'F19b_2', 'F20', 'G21', 'G22', 'G23', 'G24', 'G25a_1', 'G25b_1', 'G25c_1', 'G25a_2', 'G25b_2', 'G25c_2', 'G25a_3', 'G25b_3', 'G25c_3', 'G25a_4', 'G25b_4', 'G25c_4', 'G25a_5', 'G25b_5', 'G25c_5', 'G25a_6', 'G25b_6', 'G25c_6', 'G25a_7', 'G25b_7', 'G25c_7', 'G25a_8', 'G25b_8', 'G25c_8', 'G25a_9', 'G25b_9', 'G25c_9', 'G25a_10', 'G25b_10', 'G25c_10', 'G25a_11', 'G25b_11', 'G25c_11', 'G25a_12', 'G25b_12', 'G25c_12', 'G25a_13', 'G25b_13', 'G25c_13', 'G25a_14', 'G25b_14', 'G25c_14', 'G25a_15', 'G25b_15', 'G25c_15', 'G25a_16', 'G25b_16', 'G25c_16', 'G25a_17', 'G25b_17', 'G25c_17', 'G25a_18', 'G25b_18', 'G25c_18', 'G25a_19', 'G25b_19', 'G25c_19', 'G25a_20', 'G25b_20', 'G25c_20', 'G25a_21', 'G25b_21', 'G25c_21', 'G25a_22', 'G25b_22', 'G25c_22', 'G25a_23', 'G25b_23', 'G25c_23', 'G25a_24', 'G25b_24', 'G25c_24', 'G25a_25', 'G25b_25', 'G25c_25', 'G25a_26', 'G25b_26', 'G25c_26', 'G25a_27', 'G25b_27', 'G25c_27', 'G25a_28', 'G25b_28', 'G25c_28', 'G25a_29', 'G25b_29', 'G25c_29', 'G25a_30', 'G25b_30', 'G25c_30', 'G25a_31', 'G25b_31', 'G25c_31', 'G25a_32', 'G25b_32', 'G25c_32', 'H26_1', 'H26_2', 'H26_3', 'H27', 'I28_1', 'I28_2', 'I29_1', 'I29_2', 'K32', 'K33', 'K34', 'K35', 'K36', 'K37', 'K38', 'K38a', 'L39_1', 'L39_2', 'L39_3', 'L39_4', 'L39_5', 'L40', 'L41', 'L42_1', 'L42_2', 'L42_3', 'L42_4', 'L42_5', 'L42_6', 'L42_7', 'L43', 'L44', 'L45', 'L46', 'M47_1', 'M47_2', 'M47_3', 'M47_4', 'M47_5', 'M47_6', 'M47_7', 'M47_8', 'M47_9', 'M47_10', 'M47_11', 'M47_12', 'M47a', 'M47b', 'N48', 'N49', 'N50', 'N51', 'N52', 'N53', 'N54', 'N55_1', 'N55_2', 'N55_3', 'N56_1', 'N56_2', 'N56_3', 'N57', 'N58_1', 'N58_2', 'N58_3', 'N59', 'N60_1', 'N60_2', 'N60_3', 'N60_4', 'N60a_1', 'N60a_2', 'N60a_3', 'N60a_4', 'N61_1', 'N61_2', 'N61_3', 'N61_4', 'N61_5', 'N62', 'N62b', 'N63_1', 'N63_2', 'N63_3', 'N63_4', 'N63_5', 'N63_6', 'N63_7', 'N63_8', 'N63_9', 'N63_10', 'N64', 'N65_1', 'N65_2', 'N65_3', 'N65_4', 'N65_5', 'N65_6', 'N65_7', 'N65_8', 'O66', 'O67', 'O69', 'O70', 'O71', 'O72', 'O73', 'O74', 'O75', 'O76', 'O77', 'O78', 'O79', 'O80_1', 'O80_2', 'O80_3', 'O81_1', 'O81_2', 'O81_3', 'O82_1', 'O82_2', 'O83', 'O84_1', 'O84_2', 'O84_3', 'O84_4', 'O84_5', 'O84_6', 'O84_7', 'O84_8', 'O84_9', 'P85', 'P86', 'P87_1a', 'P87_1b', 'P87_2a', 'P87_2b', 'P87_3a', 'P87_3b', 'P87_4a', 'P87_4b', 'P87_5a', 'P87_5b', 'P87_6a', 'P87_6b', 'P87_7a', 'P87_7b', 'P87_8a', 'P87_8b', 'P87_9a', 'P87_9b', 'P87_10a', 'P87_10b', 'P87_11a', 'P87_11b', 'P87_12a', 'P87_12b', 'P87_13a', 'P87_13b', 'P87_14a', 'P87_14b', 'P87_15a', 'P87_15b', 'P87_16a', 'P87_16b', 'P87_17a', 'P87_17b', 'P87_18a', 'P87_18b', 'P87_19a', 'P87_19b', 'P87_20a', 'P87_20b', 'P87_21a', 'P87_21b', 'P87_22a', 'P87_22b', 'P87_23a', 'P87_23b', 'Q88', 'Q89', 'Q90', 'Q91', 'Q92', 'Q93', 'Q94', 'Q95', 'Q96', 'Q97', 'Q98', 'Q99', 'Q100', 'Q101', 'Q102', 'Q103', 'Q104', 'Q105', 'R106', 'R107', 'R108_1', 'R108_2', 'R108_3', 'R108_4', 'S109', 'S110', 'T111', 'T112', 'T113', 'T114_1', 'T114_2', 'T115', 'T116_1', 'T116_2', 'T117', 'T118_1', 'T118_2', 'T119_1', 'T119_2', 'U120_1', 'U120_1a', 'U120_2', 'U120_3', 'U120_4', 'U120_5', 'U120_6', 'U120_7', 'U120_7a', 'U120_8', 'U120_9', 'U120_10', 'U120_11', 'U120_12', 'U120_13', 'U120_14', 'U120_15', 'U120_15a', 'U120FZ', 'U120CANTFZ', 'U2_120F', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'W127', 'W128Cer', 'W128Cer_1', 'W128Cer_2', 'W128Cer_3', 'W128Cer_4', 'W128Cer_5', 'W128Cer_6', 'W128Cer_7', 'W128Vin', 'W128Vin_1', 'W128Vin_2', 'W128Vin_3', 'W128Vin_4', 'W128Vin_5', 'W128Vin_6', 'W128Vin_7', 'W128Vermut', 'W128Vermut_1', 'W128Vermut_2', 'W128Vermut_3', 'W128Vermut_4', 'W128Vermut_5', 'W128Vermut_6', 'W128Vermut_7', 'W128Lic', 'W128Lic_1', 'W128Lic_2', 'W128Lic_3', 'W128Lic_4', 'W128Lic_5', 'W128Lic_6', 'W128Lic_7', 'W128Comb', 'W128Comb_1', 'W128Comb_2', 'W128Comb_3', 'W128Comb_4', 'W128Comb_5', 'W128Comb_6', 'W128Comb_7', 'W128Sidra', 'W128Sidra_1', 'W128Sidra_2', 'W128Sidra_3', 'W128Sidra_4', 'W128Sidra_5', 'W128Sidra_6', 'W128Sidra_7', 'W129', 'X130_1', 'X130_2', 'X130_3', 'X130_4', 'X130_5', 'X130_6', 'X130_7', 'X130_8', 'X130_9', 'X130_10', 'X130_11', 'Y133', 'Y134', 'Y135', 'FACTORADULTO', 'CLASE_PR', 'IMCa', 'CMD1', 'CMD2', 'CMD3']
    return dataset


def create_datasets_compatible(dataset):
    indexm_array = []
    indexp_array = []
    ccaam_array = []
    ccaap_array = []
    edadm_array = []
    edadp_array = []
    mforeign_array = []
    fforeign_array = []
    mimmi_array = []
    fimmi_array = []
    estudiom_array = []
    estudiop_array = []
    profm_array = []
    profp_array = []
    casadam_array = []
    casadop_array = []
    ecivm_array = []
    ecivp_array = []
    convivenm_array = []
    convivenp_array = []

    for index, row in dataset.iterrows():

        if(row['SEXOa'] == 2):      # Mujer

            indexm_array.append(index)

            # ccaam
            ccaam_array.append(row['CCAA'])

            # edadm
            edadm_array.append(row['EDADa'])

            # mforeign
            if (row['E2_1b'] == 2):
                mforeign_array.append(0)
            else:
                mforeign_array.append(row['E2_1b'])

            # mimmi
            if (row['E1_1'] == 1):
                mimmi_array.append(0)
            elif (row['E1_1'] == 2):
                mimmi_array.append(1)
            else:
                mimmi_array.append(np.nan)

            # estudiom
            if (row['NIVEST'] == 2):
                estudiom_array.append(1)
            elif (row['NIVEST'] == 3):
                estudiom_array.append(2)
            elif (row['NIVEST'] == 4):
                estudiom_array.append(3)
            elif (row['NIVEST'] == 5):
                estudiom_array.append(4)
            elif (row['NIVEST'] == 6):
                estudiom_array.append(5)
            elif (row['NIVEST'] == 7):
                estudiom_array.append(6)
            elif (row['NIVEST'] == 8):
                estudiom_array.append(8)
            elif (row['NIVEST'] == 9):
                estudiom_array.append(9)
            else:
                estudiom_array.append(np.nan)

            # profm
            if (row['F19a_2'] in [1,2,3,4,5,6,7,8,9]):
                profm_array.append(row['F19a_2'])
            elif (row['F19a_2'] == 0):
                profm_array.append(10)
            elif (row['ACTIVa'] == 2):
                profm_array.append(11)
            elif (row['ACTIVa'] == 4):
                profm_array.append(12)
            elif (row['ACTIVa'] == 3):
                profm_array.append(13)
            elif (row['ACTIVa'] == 5):
                profm_array.append(14)
            elif (row['ACTIVa'] == 7):
                profm_array.append(15)
            else:
                profm_array.append(np.nan)

            # casadam
            if (row['E4b'] == 2):
                casadam_array.append(1)
            elif (row['E4b'] in [1,3,4,5]):
                casadam_array.append(0)
            else:
                casadam_array.append(np.nan)

            # ecivm
            if (row['E4b'] == 2):
                ecivm_array.append(0)
            elif (row['E4b'] == 1):
                ecivm_array.append(1)
            elif (row['E4b'] in [4,5]):
                ecivm_array.append(2)
            elif (row['E4b'] == 3):
                ecivm_array.append(3)
            else:
                ecivm_array.append(np.nan)

            # convivenm
            if (row['E4'] in [1,2]):
                convivenm_array.append(1)
            elif (row['E4'] == 3):
                convivenm_array.append(0)
            else:
                convivenm_array.append(np.nan)

        elif(row['SEXOa'] == 1):    # Hombre

            indexp_array.append(index)

            # ccaam
            ccaap_array.append(row['CCAA'])

            # edadp
            edadp_array.append(row['EDADa'])

            # fforeign
            if (row['E2_1b'] == 2):
                fforeign_array.append(0)
            else:
                fforeign_array.append(row['E2_1b'])

            # fimmi
            if (row['E1_1'] == 1):
                fimmi_array.append(0)
            elif (row['E1_1'] == 2):
                fimmi_array.append(1)
            else:
                fimmi_array.append(np.nan)

            # estudiop
            if (row['NIVEST'] == 2):
                estudiop_array.append(1)
            elif (row['NIVEST'] == 3):
                estudiop_array.append(2)
            elif (row['NIVEST'] == 4):
                estudiop_array.append(3)
            elif (row['NIVEST'] == 5):
                estudiop_array.append(4)
            elif (row['NIVEST'] == 6):
                estudiop_array.append(5)
            elif (row['NIVEST'] == 7):
                estudiop_array.append(6)
            elif (row['NIVEST'] == 8):
                estudiop_array.append(8)
            elif (row['NIVEST'] == 9):
                estudiop_array.append(9)
            else:
                estudiop_array.append(np.nan)

            # profp
            if (row['F19a_2'] in [1, 2, 3, 4, 5, 6, 7, 8, 9]):
                profp_array.append(row['F19a_2'])
            elif (row['F19a_2'] == 0):
                profp_array.append(10)
            elif (row['ACTIVa'] == 2):
                profp_array.append(11)
            elif (row['ACTIVa'] == 4):
                profp_array.append(12)
            elif (row['ACTIVa'] == 3):
                profp_array.append(13)
            elif (row['ACTIVa'] == 5):
                profp_array.append(14)
            elif (row['ACTIVa'] == 7):
                profp_array.append(15)
            else:
                profp_array.append(np.nan)

            # casadop
            if (row['E4b'] == 2):
                casadop_array.append(1)
            elif (row['E4b'] in [1, 3, 4, 5]):
                casadop_array.append(0)
            else:
                casadop_array.append(np.nan)

            # ecivp
            if (row['E4b'] == 2):
                ecivp_array.append(0)
            elif (row['E4b'] == 1):
                ecivp_array.append(1)
            elif (row['E4b'] in [4, 5]):
                ecivp_array.append(2)
            elif (row['E4b'] == 3):
                ecivp_array.append(3)
            else:
                ecivp_array.append(np.nan)

            # convivenp
            if (row['E4'] in [1, 2]):
                convivenp_array.append(1)
            elif (row['E4'] == 3):
                convivenp_array.append(0)
            else:
                convivenp_array.append(np.nan)

    datasetm = pd.DataFrame({'ccaam': ccaam_array, 'edadm': edadm_array, 'mforeign': mforeign_array, 'mimmi': mimmi_array,
                             'estudiom': estudiom_array,'profm': profm_array, 'casadam': casadam_array, 'ecivm': ecivm_array,
                             'convivenm': convivenm_array}, index=indexm_array)
    datasetp = pd.DataFrame({'ccaap': ccaap_array, 'edadp': edadp_array, 'fforeign': fforeign_array, 'fimmi': fimmi_array,
                             'estudiop': estudiop_array, 'profp': profp_array, 'casadop': casadop_array, 'ecivp': ecivp_array,
                             'convivenp': convivenp_array}, index=indexp_array)
    datasetm = datasetm.dropna()
    datasetp = datasetp.dropna()
    return datasetm, datasetp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create the ENSE dataset, and datasets with features compatibles with Perinatal dataset. Usage example: $python ense_dataset_creation.py dataENSE2017.txt -o pathTo/ENSE')
    parser.add_argument("input_data", help="Path to file with input data: 'dataENSE2017.txt'")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for created datasets. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_data = args['input_data']
    output_dir = args['output_dir']

    print('\nReading ENSE data from: ' + str(input_data))

    print("\nCreating ENSE dataset...")
    dataset = create_dataset(input_data)
    print("Saving created dataset in: " + str(os.path.join(output_dir, "dataENSE2017.csv")))
    dataset.to_csv(os.path.join(output_dir, "dataENSE2017.csv"), index=False)

    print("\nCreating ENSE datasets compatible with Perinatal dataset...")
    dataset_compatible_m, dataset_compatible_p = create_datasets_compatible(dataset)
    print("Saving created dataset (women) in: " + str(os.path.join(output_dir, "dataENSE2017_compatible_m.csv")))
    dataset_compatible_m.to_csv(os.path.join(output_dir, "dataENSE2017_compatible_m.csv"))
    print("Saving created dataset (men) in: " + str(os.path.join(output_dir, "dataENSE2017_compatible_p.csv")))
    dataset_compatible_p.to_csv(os.path.join(output_dir, "dataENSE2017_compatible_p.csv"))





