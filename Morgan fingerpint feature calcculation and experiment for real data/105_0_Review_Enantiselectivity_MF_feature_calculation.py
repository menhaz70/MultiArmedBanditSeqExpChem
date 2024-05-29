
import matplotlib.pyplot as plt
import CIMtools 
from os import environ

from CGRtools import RDFRead, ReactionContainer, SDFRead, SMILESRead, smiles
import pandas as pd
import numpy as np
from collections.abc import Mapping


from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt 
import time       
import random
from random import randint 
from operator import itemgetter   ### to short the rewards
from sklearn.preprocessing import StandardScaler
import shutil
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from rdkit.Chem.rdmolfiles import SDWriter
from scipy.stats import pearsonr
from molvs import standardize_smiles






data1 = pd.read_excel(r"THF_THP_data.xls", sheet_name='THF')
data2 = pd.read_excel(r"THF_THP_data.xls", sheet_name='THP')
data3 = pd.read_excel(r"THF_THP_data.xls", sheet_name='External validation')

smiles_THF = data1["SMILES"]
smiles_THP = data2["SMILES"]
smiles_ext = data3["SMILES"]

ddg1 = data1["ddG"]
ddg2 = data2["ddG"]
ddg3 = data3["ddG"]

# 70 data
smiles = pd.concat([smiles_THF, smiles_THP], ignore_index=True)
ddg = pd.concat([ddg1, ddg2], ignore_index=True)

def read_morgan_fps(smiles, ddg):
    
    morgan_fns= []
    for i in range(len(smiles)):
        stan_smiles = standardize_smiles(smiles[i])
        mol = Chem.MolFromSmiles(stan_smiles, sanitize=True)
        fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
        non_zero_position = tuple(fps.GetOnBits())
        list = [0] * fps.GetNumBits()
        for j in non_zero_position:
            list[j] = 1       
        morgan_fns.append(list)        
        
    fn_matrix = pd.DataFrame()
    for i in range(len(morgan_fns)):
        fn_matrix[i] = morgan_fns[i]        

    fns = []
    for i in range(len(fn_matrix)):
        elements = ("fn_%s" % i)
        fns.append(elements)    

    feature = fn_matrix.T
    feature.columns = fns  

    data = pd.concat([ddg, feature], axis=1)
    data.to_csv("THF_THP_feature_MF.csv") 
    
    
    return data
    
    
feature = read_morgan_fps(smiles, ddg)











