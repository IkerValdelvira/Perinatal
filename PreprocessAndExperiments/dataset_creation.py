import argparse
import math
import os

import pandas as pd
import numpy as np


def create_dataset(data, starting_index=0, split_num=1):

    # Variable 'multipli'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'multipli'...")
    multipli_array = data['multipli'].to_numpy()
    dataset = pd.DataFrame(multipli_array, columns=['multipli'])

    # Inicializar el indice del nuevo dataframe
    dataset.index += starting_index

    # Variable 'nacvn'
    # Clasificación de 'nacvn':
    #   - 0: 2 (muerto)
    #   - 1: 1 (vivo)
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'nacvn'...")
    nacvm_array = []
    for index, row in data.iterrows():
        if (row['nacvn'] == 2):
            nacvm_array.append(0)
        elif (row['nacvn'] == 1):
            nacvm_array.append(1)
    dataset['nacvn'] = nacvm_array

    # Variable 'v24hn'
    # Clasificación de 'v24hn':
    #   - 0: 2 (no)
    #   - 1: 1 (sí)
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'v24hn'...")
    v24hn_array = []
    for index, row in data.iterrows():
        if (row['v24hn'] == 2):
            v24hn_array.append(0)
        elif (row['v24hn'] == 1):
            v24hn_array.append(1)
        elif (math.isnan(row['v24hn'])):
            v24hn_array.append(np.nan)
    dataset['v24hn'] = v24hn_array

    # Variable 'numhv'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'numhv'...")
    numhv_array = data['numhv'].to_numpy()
    dataset['numhv'] = numhv_array

    # Variable 'firstborn'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'firstborn'...")
    firstborn_array = data['firstborn'].to_numpy()
    dataset['firstborn'] = list(map(int, firstborn_array))

    # Variable 'singleton'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'singleton'...")
    singleton_array = data['singleton'].to_numpy()
    dataset['singleton'] = singleton_array

    # Variable 'propar'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'propar'...")
    propar_array = data['propar'].to_numpy()
    dataset['propar'] = propar_array

    # Variable 'mespar'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'mespar'...")
    mespar_array = data['mespar'].to_numpy()
    dataset['mespar'] = mespar_array

    # Variable 'anopar'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'anopar'...")
    anopar_array = data['anopar'].to_numpy()
    dataset['anopar'] = anopar_array

    # Variables 'cesarea', 'cesarean'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'cesarea'...")
    cesarea_array = data['cesarean'].to_numpy()
    dataset['cesarea'] = cesarea_array

    # Variables 'sexo', 'female'
    # Clasificación de 'sexo':
    #   - 0: 6 (female)
    #   - 1: 1 (male)
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'sexo'...")
    sexo_array = []
    for index, row in data.iterrows():
        if (row['sexo'] == 6):
            sexo_array.append(0)
        elif (row['sexo'] == 1):
            sexo_array.append(1)
    dataset['sexo'] = sexo_array

    # Variable 'peson'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'peson'...")
    peson_array = data['peson'].to_numpy()
    dataset['peson'] = peson_array

    # Variable 'semanas'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'semanas'...")
    semanas_array = data['semanas'].to_numpy()
    dataset['semanas'] = semanas_array

    # Variable 'gestage3'
    # Clasificación de 'gestage3':
    #   - 0: “Pre-term (<37)”
    #   - 1: “Term & late-term (37-41)”
    #   - 2: ”Post-term(42+)”
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'gestage3'...")
    gestage3_array = []
    for index, row in data.iterrows():
        gestage3 = row['gestage3']
        if (gestage3 == 'Pre-term (<37)'):
            gestage3_array.append(0)
        elif (gestage3 == 'Term & late-term (37-41)'):
            gestage3_array.append(1)
        elif (gestage3 == 'Post-term(42+)'):
            gestage3_array.append(2)
        else:
            gestage3_array.append(math.nan)
    dataset['gestage3'] = gestage3_array

    # Variable 'gestage4'
    # Clasificación de 'gestage4':
    #   - 0: “Very pre-term (<32)”
    #   - 1: “Mod/late pre-term (32-37)”
    #   - 2: “Term & late-term (37-41)”
    #   - 3: ”Post-term(42+)”
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'gestage4'...")
    gestage4_array = []
    for index, row in data.iterrows():
        gestage4 = row['gestage4']
        if (gestage4 == 'Very pre-term (<32)'):
            gestage4_array.append(0)
        elif (gestage4 == 'Mod/late pre-term (32-37)'):
            gestage4_array.append(1)
        elif (gestage4 == 'Term & late-term (37-41)'):
            gestage4_array.append(2)
        elif (gestage4 == 'Post-term(42+)'):
            gestage4_array.append(3)
        else:
            gestage4_array.append(math.nan)
    dataset['gestage4'] = gestage4_array

    # Variable 'premature'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'premature'...")
    semanas_array = data['premature'].to_numpy()
    dataset['premature'] = semanas_array

    # Variable 'normterm'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'normterm'...")
    normterm_array = data['normterm'].to_numpy()
    dataset['normterm'] = normterm_array

    # Variable 'postterm'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'postterm'...")
    postterm_array = data['postterm'].to_numpy()
    dataset['postterm'] = postterm_array

    # Variable 'vpreterm'
    print("\n[SPLIT_" + str(split_num) + "] Creating variable 'vpreterm'...")
    vpreterm_array = data['vpreterm'].to_numpy()
    dataset['vpreterm'] = list(map(int, vpreterm_array))

    # Variable 'edadm'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'edadm'...")
    edadm_array = data['edadm'].to_numpy()
    dataset['edadm'] = edadm_array

    # Variable 'edadm6'
    # Clasificación de 'edadm6':
    #   - 0: “<20”
    #   - 1: “20-24”
    #   - 2: “25-29”
    #   - 3: “30-34”
    #   - 4: “35-39”
    #   - 5: “c”
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'edadm6'...")
    edadm6_array = []
    for index, row in data.iterrows():
        if (row['edadm6'] == '<20'):
            edadm6_array.append(0)
        elif (row['edadm6'] == '20-24'):
            edadm6_array.append(1)
        elif (row['edadm6'] == '25-29'):
            edadm6_array.append(2)
        elif (row['edadm6'] == '30-34'):
            edadm6_array.append(3)
        elif (row['edadm6'] == '35-39'):
            edadm6_array.append(4)
        elif (row['edadm6'] == '40+'):
            edadm6_array.append(5)
    dataset['edadm6'] = edadm6_array

    # Variable 'edadm35'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'edadm35'...")
    edadm35_array = data['edadm35'].to_numpy()
    dataset['edadm35'] = list(map(int, edadm35_array))

    # Variable 'edadp'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'edadp'...")
    edadp_array = data['edadp'].to_numpy()
    dataset['edadp'] = edadp_array

    # Variable 'edadp35'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'edadp35'...")
    edadp35_array = data['edadp35'].to_numpy()
    dataset['edadp35'] = edadp35_array

    # Variables 'mforeign', 'nacioem'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'mforeign'...")
    mforeign_array = data['mforeign'].to_numpy()
    dataset['mforeign'] = mforeign_array

    # Variables 'fforeign', 'nacioep'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'fforeign'...")
    fforeign_array = data['fforeign'].to_numpy()
    dataset['fforeign'] = fforeign_array

    # Variables 'nacmad', 'paisnacm'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'paisnacm'...")
    paisnacm_array = []
    for index, row in data.iterrows():
        ano = row['anopar']
        if (1996 <= ano <= 2006):
            if (math.isnan(row['nacmad'])):
                if (dataset.loc[index]['mforeign'] == 0):       # Si la nacionalidad no es extranjera, pais de nacionalidad = 108 (España)
                    paisnacm_array.append(108)
                else:
                    paisnacm_array.append(np.nan)
            else:
                paisnacm_array.append(row['nacmad'])
        elif (2007 <= ano <= 2019):
            if(math.isnan(row['paisnacm'])):
                if (dataset.loc[index]['mforeign'] == 0):       # Si la nacionalidad no es extranjera, pais de nacionalidad = 108 (España)
                    paisnacm_array.append(108)
                else:
                    paisnacm_array.append(np.nan)
            else:
                paisnacm_array.append(row['paisnacm'])
    dataset['paisnacm'] = paisnacm_array

    # Variables 'nacpad', 'paisnacp'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'paisnacp'...")
    paisnacp_array = []
    for index, row in data.iterrows():
        ano = row['anopar']
        if (1996 <= ano <= 2006):
            if (math.isnan(row['nacpad'])):
                if(dataset.loc[index]['fforeign'] == 0):         # Si la nacionalidad no es extranjera, pais de nacionalidad = 108 (España)
                    paisnacp_array.append(108)
                else:
                    paisnacp_array.append(np.nan)
            else:
                paisnacp_array.append(row['nacpad'])
        elif (2007 <= ano <= 2019):
            if (math.isnan(row['paisnacp'])):
                if (dataset.loc[index]['fforeign'] == 0):       # Si la nacionalidad no es extranjera, pais de nacionalidad = 108 (España)
                    paisnacp_array.append(108)
                else:
                    paisnacp_array.append(np.nan)
            else:
                paisnacp_array.append(row['paisnacp'])
    dataset['paisnacp'] = paisnacp_array

    # Variable 'mimmi'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'mimmi'...")
    mimmi_array = data['mimmi'].to_numpy()
    dataset['mimmi'] = mimmi_array

    # Variable 'fimmi'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'fimmi'...")
    fimmi_array = data['fimmi'].to_numpy()
    dataset['fimmi'] = fimmi_array

    # Variable 'paisnxm'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'paisnxm'...")
    paisnxm_array = []
    for index, row in data.iterrows():
        if(math.isnan(row['paisnxm'])):
            if (dataset.loc[index]['mimmi'] == 0):      # Si el origen es español, pais de origen = 108 (España)
                paisnxm_array.append(108)
            else:
                paisnxm_array.append(np.nan)
        else:
            paisnxm_array.append(row['paisnxm'])
    dataset['paisnxm'] = paisnxm_array

    # Variable 'paisnxp'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'paisnxp'...")
    paisnxp_array = []
    for index, row in data.iterrows():
        if (math.isnan(row['paisnxp'])):
            if (dataset.loc[index]['fimmi'] == 0):      # Si el origen es español, pais de origen = 108 (España)
                paisnxp_array.append(108)
            else:
                paisnxp_array.append(np.nan)
        else:
            paisnxp_array.append(row['paisnxp'])
    dataset['paisnxp'] = paisnxp_array

    # Variable 'estudiom'
    # * Ver nueva clasificación creada en la documentación
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'estudiom'...")
    estudiom_array = []
    for index, row in data.iterrows():
        ano = row['anopar']
        if (1996 <= ano <= 2006):
            estudiom_array.append(np.nan)   # Todos valor NaN
        elif (2007 <= ano <= 2015):
            estudiom = row['estudiom']
            if (estudiom == 0 or math.isnan(estudiom)):
                estudiom_array.append(np.nan)
            elif (estudiom in [1, 2, 3, 4, 5, 6, 7, 8, 9]):
                estudiom_array.append(estudiom)
            elif (estudiom == 10):
                estudiom_array.append(12)
        elif (2016 <= ano <= 2019):
            estudiom = row['estudiom']
            if (estudiom == 99 or math.isnan(estudiom)):
                estudiom_array.append(np.nan)
            elif (estudiom in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
                estudiom_array.append(estudiom)
    dataset['estudiom'] = estudiom_array

    # Variable 'estudiop'
    # * Ver nueva clasificación creada en la documentación
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'estudiop'...")
    estudiop_array = []
    for index, row in data.iterrows():
        ano = row['anopar']
        if (1996 <= ano <= 2006):
            estudiop_array.append(np.nan)  # Todos valor NaN
        elif (2007 <= ano <= 2015):
            estudiop = row['estudiop']
            if (estudiop in [0, 12, 80] or math.isnan(estudiop)):
                estudiop_array.append(np.nan)
            elif (estudiop in [1, 2, 3, 4, 5, 6, 7, 8, 9]):
                estudiop_array.append(estudiop)
            elif (estudiop == 10):
                estudiop_array.append(12)
        elif (2016 <= ano <= 2019):
            estudiop = row['estudiop']
            if (estudiop == 99 or math.isnan(estudiop)):
                estudiop_array.append(np.nan)
            elif (estudiop in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
                estudiop_array.append(estudiop)
    dataset['estudiop'] = estudiop_array

    # Variable 'educm'
    # Clasificación de 'educm':
    #   - 0: “Primary or less”
    #   - 1: “Secondary”
    #   - 2: “University”
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'educm'...")
    educm_array = []
    for index, row in data.iterrows():
        if (row['educm'] == 'Primary or less'):
            educm_array.append(0)
        elif (row['educm'] == 'Secondary'):
            educm_array.append(1)
        elif (row['educm'] == 'University'):
            educm_array.append(2)
        elif (math.isnan(float(row['educm']))):
            educm_array.append(np.nan)
    dataset['educm'] = educm_array

    # Variable 'educp'
    # Clasificación de 'educp':
    #   - 0: “Primary or less”
    #   - 1: “Secondary”
    #   - 2: “University”
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'educp'...")
    educp_array = []
    for index, row in data.iterrows():
        if (row['educp'] == 'Primary or less'):
            educp_array.append(0)
        elif (row['educp'] == 'Secondary'):
            educp_array.append(1)
        elif (row['educp'] == 'University'):
            educp_array.append(2)
        else:
            educp_array.append(np.nan)
    dataset['educp'] = educp_array

    # Variables 'profm', 'cautom', 'relam'
    # * Ver nueva clasificación creada en la documentación
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'profm'...")
    profm_array = []
    for index, row in data.iterrows():
        ano = row['anopar']
        if (1996 <= ano <= 2006):
            prof = row['profm']
            if (prof == 12 or math.isnan(prof)):
                profm_array.append(np.nan)
            elif (prof == 2):
                profm_array.append(1)
            elif (prof == 1):
                profm_array.append(2)
            elif (prof == 3):
                profm_array.append(4)
            elif (prof == 4 or prof == 5):
                profm_array.append(5)
            elif (prof == 6):
                profm_array.append(6)
            elif (prof == 7):
                profm_array.append(7)
            elif (prof == 8):
                profm_array.append(10)
            elif (prof == 10):
                profm_array.append(11)
            elif (prof == 9):
                profm_array.append(12)
            elif (prof == 11):
                profm_array.append(3)
        elif (2007 <= ano <= 2015):
            prof = row['cautom']
            if (prof == 0 or math.isnan(prof)):
                profm_array.append(np.nan)
            elif (prof == 2):
                profm_array.append(1)
            elif (prof == 3):
                profm_array.append(2)
            elif (prof == 4):
                profm_array.append(3)
            elif (prof == 5):
                profm_array.append(4)
            elif (prof == 6):
                profm_array.append(5)
            elif (prof == 7):
                profm_array.append(6)
            elif (prof == 8):
                profm_array.append(7)
            elif (prof == 9):
                profm_array.append(8)
            elif (prof == 10):
                profm_array.append(9)
            elif (prof == 1):
                profm_array.append(10)
            elif (prof == 12):
                profm_array.append(11)
            elif (prof == 11):
                profm_array.append(12)
            elif (prof == 13):
                profm_array.append(13)
        elif (2016 <= ano <= 2019):
            prof1 = row['cautom']
            prof2 = row['relam']
            if (prof1 == 99 or math.isnan(prof1)):
                if (prof2 in [0, 1, 3, 9] or math.isnan(prof2)):
                    profm_array.append(np.nan)
                elif (prof2 == 2):
                    profm_array.append(11)
                elif (prof2 == 7):
                    profm_array.append(12)
                elif (prof2 == 5):
                    profm_array.append(13)
                elif (prof2 == 4):
                    profm_array.append(14)
                elif (prof2 == 6):
                    profm_array.append(15)
            elif (prof1 == 1):
                profm_array.append(1)
            elif (prof1 == 2):
                profm_array.append(2)
            elif (prof1 == 3):
                profm_array.append(3)
            elif (prof1 == 4):
                profm_array.append(4)
            elif (prof1 == 5):
                profm_array.append(5)
            elif (prof1 == 6):
                profm_array.append(6)
            elif (prof1 == 7):
                profm_array.append(7)
            elif (prof1 == 8):
                profm_array.append(8)
            elif (prof1 == 9):
                profm_array.append(9)
            elif (prof1 == 00):
                profm_array.append(10)
    dataset['profm'] = profm_array

    # Variables 'profp', 'cautop', 'relap'
    # * Ver nueva clasificación creada en la documentación
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'profp'...")
    profp_array = []
    for index, row in data.iterrows():
        ano = row['anopar']
        if (1996 <= ano <= 2006):
            prof = row['profp']
            if (prof == 12 or math.isnan(prof)):
                profp_array.append(np.nan)
            elif (prof == 2):
                profp_array.append(1)
            elif (prof == 1):
                profp_array.append(2)
            elif (prof == 3):
                profp_array.append(4)
            elif (prof == 4 or prof == 5):
                profp_array.append(5)
            elif (prof == 6):
                profp_array.append(6)
            elif (prof == 7):
                profp_array.append(7)
            elif (prof == 8):
                profp_array.append(10)
            elif (prof == 10):
                profp_array.append(11)
            elif (prof == 9):
                profp_array.append(12)
            elif (prof == 11):
                profp_array.append(3)
        elif (2007 <= ano <= 2015):
            prof = row['cautop']
            if (prof == 0 or math.isnan(prof)):
                profp_array.append(np.nan)
            elif (prof == 2):
                profp_array.append(1)
            elif (prof == 3):
                profp_array.append(2)
            elif (prof == 4):
                profp_array.append(3)
            elif (prof == 5):
                profp_array.append(4)
            elif (prof == 6):
                profp_array.append(5)
            elif (prof == 7):
                profp_array.append(6)
            elif (prof == 8):
                profp_array.append(7)
            elif (prof == 9):
                profp_array.append(8)
            elif (prof == 10):
                profp_array.append(9)
            elif (prof == 1):
                profp_array.append(10)
            elif (prof == 12):
                profp_array.append(11)
            elif (prof == 11):
                profp_array.append(12)
            elif (prof == 13):
                profp_array.append(13)
        elif (2016 <= ano <= 2019):
            prof1 = row['cautop']
            prof2 = row['relap']
            if (prof1 == 99 or math.isnan(prof1)):
                if (prof2 in [0, 1, 3, 9] or math.isnan(prof2)):
                    profp_array.append(np.nan)
                elif (prof2 == 2):
                    profp_array.append(11)
                elif (prof2 == 7):
                    profp_array.append(12)
                elif (prof2 == 5):
                    profp_array.append(13)
                elif (prof2 == 4):
                    profp_array.append(14)
                elif (prof2 == 6):
                    profp_array.append(15)
            elif (prof1 == 1):
                profp_array.append(1)
            elif (prof1 == 2):
                profp_array.append(2)
            elif (prof1 == 3):
                profp_array.append(3)
            elif (prof1 == 4):
                profp_array.append(4)
            elif (prof1 == 5):
                profp_array.append(5)
            elif (prof1 == 6):
                profp_array.append(6)
            elif (prof1 == 7):
                profp_array.append(7)
            elif (prof1 == 8):
                profp_array.append(8)
            elif (prof1 == 9):
                profp_array.append(9)
            elif (prof1 == 00):
                profp_array.append(10)
    dataset['profp'] = profp_array

    # Variable 'occupm'
    # Clasificación de 'occupm':
    #   - 0: “Inactive”
    #   - 1: “Low”
    #   - 2: “High/mid”
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'occupm'...")
    occupm_array = []
    for index, row in data.iterrows():
        if (row['occupm'] == 'Inactive'):
            occupm_array.append(0)
        elif (row['occupm'] == 'Low'):
            occupm_array.append(1)
        elif (row['occupm'] == 'High/mid'):
            occupm_array.append(2)
        elif (math.isnan(float(row['occupm']))):
            occupm_array.append(np.nan)
    dataset['occupm'] = occupm_array

    # Variable 'occupp'
    # Clasificación de 'occupp':
    #   - 0: “Low/inactive”
    #   - 1: “High/mid”
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'occupp'...")
    occupp_array = []
    for index, row in data.iterrows():
        if (row['occupp'] == 'Low/inactive'):
            occupp_array.append(0)
        elif (row['occupp'] == 'High/mid'):
            occupp_array.append(1)
        elif (math.isnan(float(row['occupp']))):
            occupp_array.append(np.nan)
    dataset['occupp'] = occupp_array

    # Variables 'cas', 'married'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'casada'...")
    casada_array = data['married'].to_numpy()
    dataset['casada'] = list(map(int, casada_array))

    # Variable 'ecivm'
    # Clasificación de 'ecivm':
    #   - 0: 1 (casada)
    #   - 1: 2 (soltera)
    #   - 2: 3 (separada/divorciada)
    #   - 3: 4 (viuda)
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'ecivm'...")
    ecivm_array = []
    for index, row in data.iterrows():
        if (row['ecivm'] == 1):
            ecivm_array.append(0)
        elif (row['ecivm'] == 2):
            ecivm_array.append(1)
        elif (row['ecivm'] == 3):
            ecivm_array.append(2)
        elif (row['ecivm'] == 4):
            ecivm_array.append(3)
        elif (math.isnan(row['ecivm'])):
            if(dataset.loc[index]['casada'] == 0):
                ecivm_array.append(np.nan)      # No está casada, pero no se sabe si está soltera, separada/divorciada o viuda
            elif(dataset.loc[index]['casada'] == 1):
                ecivm_array.append(0)
    dataset['ecivm'] = ecivm_array

    # Variable 'phecho'
    # Clasificación de 'phecho':
    #   - 0: 2 (no)
    #   - 1: 1 (si)
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'phecho'...")
    phecho_array = []
    for index, row in data.iterrows():
        if (row['phecho'] == 2):
            phecho_array.append(0)
        elif (row['phecho'] == 1):
            phecho_array.append(1)
        elif (math.isnan(row['phecho'])):
            phecho_array.append(np.nan)
    dataset['phecho'] = phecho_array

    # Variable 'couple'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'pareja'...")
    pareja_array = []
    for index, row in data.iterrows():
        if (row['couple'] == 0):
            pareja_array.append(0)
        elif (row['couple'] == 1):
            pareja_array.append(1)
        elif (math.isnan(row['couple'])):
            if (dataset.loc[index]['casada'] == 1 or dataset.loc[index]['phecho'] == 1):
                pareja_array.append(1)
            else:
                pareja_array.append(np.nan)
    dataset['pareja'] = pareja_array

    # Variable 'conviven'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'conviven'...")
    conviven_array = data['conviven'].to_numpy()
    dataset['conviven'] = conviven_array

    # Variable 'mft'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'mft'...")
    mft_array = data['mft'].to_numpy()
    dataset['mft'] = list(map(int, mft_array))

    # Variable 'brank'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'brank'...")
    brank_array = data['brank'].to_numpy()
    dataset['brank'] = list(map(int, brank_array))

    # Variable 'lbw'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'lbw'...")
    lbw_array = data['lbw'].to_numpy()
    dataset['lbw'] = lbw_array

    # Variable 'nbw'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'nbw'...")
    nbw_array = data['nbw'].to_numpy()
    dataset['nbw'] = nbw_array

    # Variable 'hbw'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'hbw'...")
    hbw_array = data['hbw'].to_numpy()
    dataset['hbw'] = hbw_array

    # Variable 'pesorec'
    # Clasificación de 'pesorec':
    #   - 0: 'Bajo'
    #   - 1: 'Normal'
    #   - 2: 'Alto'
    print("\n[SPLIT_" + str(split_num) + "] Creating variables 'pesorec'...")
    pesorec_array = []
    for index, row in data.iterrows():
        if (row['pesorec'] == 'Bajo'):
            pesorec_array.append(0)
        elif (row['pesorec'] == 'Normal'):
            pesorec_array.append(1)
        elif (row['pesorec'] == 'Alto'):
            pesorec_array.append(2)
        else:
            pesorec_array.append(np.nan)
    dataset['pesorec'] = pesorec_array

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create the Perinatal dataset. Usage example: $python dataset_creation.py dataPerinatal.csv -o pathTo/Perinatal')
    parser.add_argument("input_data", help="Path to file with input data.")
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created dataset. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_data = args['input_data']
    output_dir = args['output_dir']

    print('\nReading Perinatal data from: ' + str(input_data))
    data = pd.read_csv(input_data)

    print("\nCreating data splits of 1 million items...")
    data_splits = []
    start_indices = []
    step = 1000000
    start = 0
    end = step
    while (start < data.shape[0]):
        if(end > data.shape[0]):
            end = data.shape[0]
        dataX = data.iloc[start:end, :]
        data_splits.append(dataX)
        start_indices.append(start)
        start += step
        end += step
    print("A total of " + str(len(start_indices)) + " splits have been created from the original data set.")

    datasets = []
    final_dataset = None
    for i in range(len(data_splits)):
        print("\nCREATING DATASET FOR SPLIT " + str(i+1) + "...")
        dataset = create_dataset(data_splits[i], start_indices[i], (i+1))
        print("\nSaving created dataset split in: " + str(os.path.join(output_dir, "dataPerinatal_converted_split" + str(i+1) + ".csv")))
        dataset.to_csv(os.path.join(output_dir, "dataPerinatal_converted_split" + str(i+1) + ".csv"), index=False)
        datasets.append(dataset)
        if(final_dataset is None):
            final_dataset = dataset
        else:
            final_dataset = pd.concat([final_dataset, dataset])

    """print("\nMerging all datasets created by each split...")
    dataset = pd.concat(datasets)"""
    print("Saving created final dataset in: " + str(os.path.join(output_dir, "dataPerinatal_converted.csv")))
    final_dataset.to_csv(os.path.join(output_dir, "dataPerinatal_converted.csv"), index=False)





