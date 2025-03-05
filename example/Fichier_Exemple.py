#!/usr/bin/env python
# coding: utf-8

# # Fichier Exemple : utilisation des modèles développés
# ### 1. Modèle IA pour TC
# ### 2. Modèle IA pour PC
# ### 3. Modèle IA pour ACEN
# ### 4. Modèle IA pour NBP - normal boiling point
# ### 5. Modèle IA pour TTR - point triple
# ### 6. Modèle IA pour VC
# ### Copyright - LRGP - Nancy 2024 - Roda Bounaceur
# 

# ## -- Step 1 : Appel de la classe modele et importation des différents modèles IA développés

# In[5]:
#
# Avec l'approche "Ensemble Learning", Le modèle finale est une moyenne de plusieurs sous-modèles
# La classe ainsi développée permet de faire cette moyenne automatiquement
#
from sklearn.base import BaseEstimator, RegressorMixin

class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)


# In[18]:
#
# Chargement des modèles IA
#
#
import joblib
#
TC = joblib.load('./01_modele_final_TC.joblib')      # [K]
PC = joblib.load('./02_modele_final_PC.joblib')      # [bar]
ACEN = joblib.load('./03_modele_final_ACEN.joblib')  # [-]
NBP = joblib.load('./04_modele_final_NBP.joblib')    # [K]

# In[7]:
#
# Appel de la liste des descripteurs à conserver
# Cette liste a été déterminé après l'analyse statistique de la database complète
# En se focalisant sur l'étude des TC, 247 descripteurs ont été retenus
#
# Lire les noms de colonnes d'un fichier texte dans une liste
with open('noms_colonnes_247_TC.txt', 'r') as f:
    noms_colonnes_247_TC = [ligne.strip() for ligne in f]
del noms_colonnes_247_TC[0] 
# noms_colonnes_247_TC
#


# ## -- Step 2 : Importation des modules pour Mordred et RDkit et des fonctions associées

# In[8]:
#
# Importation des bibliothèques RDKIT + Mordred
#
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import rdkit.Chem.inchi
#
from mordred import Calculator, descriptors
import mordred

#
# Ecriture des fonctions nécessaires
#
def All_Mordred_descriptors(data): # Fonction d'appelle des descripteurs
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in data]
   
    # pandas df
    df = calc.pandas(mols)
    
    #return mols # et commenter df pour tester les molecules
    return df

#
# Fonction pour obtenir la formule brute + notation InChiKey
#
def smiles_to_Inchikey_and_molecule_1(SMILES): 
    mol = Chem.MolFromSmiles(SMILES)
    smiles = Chem.MolToSmiles(mol)
    Inchikey = rdkit.Chem.inchi.MolToInchiKey(mol)
    descriptors = All_Mordred_descriptors_1(SMILES)
    nC = descriptors["nC"].iloc[0]
    nH = descriptors["nH"].iloc[0]
    nO = descriptors["nO"].iloc[0]
    nN = descriptors["nN"].iloc[0]
    return Inchikey,smiles,nC,nH,nO,nN

#
# Fonction ecriture de la notation smile canonique
#
def canonical_smiles(smiles): 
    mols = [Chem.MolFromSmiles(smi) for smi in smiles] 
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles

#
# Même fonction mais ne renvoie que l'Inchikey
# Il évite la latence due à l'appel de Mordred
#
def smiles_to_Inchikey(SMILES):
    
    mol = Chem.MolFromSmiles(SMILES)
    smiles = Chem.MolToSmiles(mol)
    Inchikey = rdkit.Chem.inchi.MolToInchiKey(mol)
    return(Inchikey)


# ## -- Step 4 : Importation des modules python de base

# In[9]:
#
# Importation des bibliothèques de bases - Pandas et Numpy - pour manipuler les data, etc ...
#
import pandas as pd  
import numpy as np
#

# ## -- Step 3 : Exemple de calcul

# In[10]:
#
# Importation d'un fichier de data smile
#
df =  pd.read_csv('Liste_Alcanes.txt',sep='*')
df


# In[13]:
#
# Calcul de tous les descripteurs possibles avec la méthode Mordred
#
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
#
df_all_descriptors = All_Mordred_descriptors(df['SMILES'])


# In[14]:
#
# Réduction du nombre de descripteurs à ceux définis par l'analyse statistique - 245 data
#
X_Alcane = df_all_descriptors[noms_colonnes_247_TC]
#


# In[15]:
#
# Nettoyage des data - A faire obligatoirement car parfois du texte subsiste ou des NaN, etc.
#
for col in X_Alcane.columns:
    # Convertir la colonne en float en remplaçant les valeurs non convertibles par zéro
    X_Alcane.loc[:, col] = pd.to_numeric(X_Alcane[col], errors='coerce').fillna(0)
#
X_Alcane = X_Alcane.fillna(0)
X_Alcane = X_Alcane.astype(float)
#
X_Alcane


# In[16]:
#
# Estimation des Propriétés Thermo
#
XX = X_Alcane
#
TC_predicted   = []
PC_predicted   = []
ACEN_predicted = []
NBP_predicted  = []
#
TC_predicted   = TC.predict( XX )
PC_predicted   = PC.predict( XX )
ACEN_predicted = ACEN.predict( XX )
NBP_predicted  = NBP.predict( XX )
#


# In[17]:
#
# Après on fait ce que l'on veut - Ici affichage simple des résultats
#
for index in range(len(TC_predicted)):
    print(TC_predicted[index] , PC_predicted[index] , ACEN_predicted[index] , NBP_predicted[index] )

