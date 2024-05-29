
import matplotlib.pyplot as plt
import CIMtools 
from os import environ
from CGRtools import RDFRead, ReactionContainer, SDFRead, SMILESRead, smiles
import pandas as pd
import numpy as np
from collections.abc import Mapping


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
smiles_ = pd.concat([smiles_THF, smiles_THP], ignore_index=True)
ddg = pd.concat([ddg1, ddg2], ignore_index=True)


# 77 data
# smiles_ = pd.concat([smiles_THF, smiles_THP, smiles_ext], ignore_index=True)
# ddg = pd.concat([ddg1, ddg2, ddg3], ignore_index=True)




molecules_ = []
for i in range(len(smiles_)):    
    mol = smiles(smiles_[i])   
    molecules_.append(mol)



from sklearn.base import BaseEstimator, TransformerMixin

class Augmentor(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0, upper=0):
        self.feature_names = []
        self.lower = lower
        self.upper = upper
    
    def fit(self, X, y=None):
        for i, mol in enumerate(X):
            for length in range(self.lower, self.upper):
                for atom in mol.atoms():
                    sub = str(mol.augmented_substructure([atom[0]], deep=length))
                    if sub not in self.feature_names:
                        self.feature_names.append(sub)
        return self
        
    def transform(self, X, y=None):
        table = pd.DataFrame(columns=self.feature_names)
        for i, mol in enumerate(X):
            table.loc[len(table)] = 0
            for sub in self.feature_names:
                mapping = list(smiles(sub).get_mapping(mol, optimize=False))
                table.loc[i,sub] = len(mapping)
        return table
    
    def get_feature_names(self):
        return self.feature_names

fragmentor = Augmentor(lower=0, upper=4)
feature = fragmentor.fit_transform(molecules_)
# data = pd.concat([ddg, feature], axis=1)

# data.to_csv("THF_THP_feature77.csv") 
