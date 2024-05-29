
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






data = pd.read_csv(r"C:\Users\menha\OneDrive - 国立大学法人 北海道大学\PhD work\Data\FreeSolv.csv")    
smiles_ = data["smiles"]   


molecules_ = []
for i in range(len(smiles_)):    
    mol = smiles(smiles_[i])   
    molecules_.append(mol)





def read_morgan_fps():
    sheet = pd.read_csv(r"C:\Users\menha\OneDrive - 国立大学法人 北海道大学\PhD work\Data\FreeSolv.csv")    
    
    smiles = sheet["smiles"] 
    

    
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

    df2 = fn_matrix.T
    # df2.columns = fns  

    df2.to_excel("freesolved_feature_MF.xlsx")
    
    return df2
    
    
df2 = read_morgan_fps()









