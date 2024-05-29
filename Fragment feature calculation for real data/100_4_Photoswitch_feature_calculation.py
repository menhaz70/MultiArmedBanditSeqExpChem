
import matplotlib.pyplot as plt
import CIMtools 
from os import environ

from CGRtools import RDFRead, ReactionContainer, SDFRead, SMILESRead, smiles
import pandas as pd
import numpy as np
from collections.abc import Mapping

data = pd.read_csv(r"C:\Users\menha\OneDrive - 国立大学法人 北海道大学\PhD work\Data\photoswitches.csv")  

extracted_data = data[["SMILES", "E isomer pi-pi* wavelength in nm"]].dropna().reset_index(drop=True)

extracted_data.to_csv("photoswitch_data.csv")


# extracted_data["E isomer pi-pi* wavelength in nm"].tolist()

smiles_ = extracted_data["SMILES"]   

# smiles_ = data["SMILES"]  


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
# data2 = pd.concat([ddg, feature], axis=1)

feature.to_excel("photoswitch_feature.xlsx")
