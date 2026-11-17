# tcPR_Calcul

Thermodynamic properties of a pure compound from its molecular structure only: AI-predicted critical properties and ideal-gas heat capacity, used as input parameters of the volume-translated Peng-Robinson equation of state (**tc-PR**).

This folder provides:

- `models/`: the trained AI models (and the descriptor list) used for the predictions;
- `example/`: a stand-alone Python script that runs the tc-PR calculation for a given molecule (1-propanol is the worked example).

## 1. Principle

From a SMILES notation, the workflow is:

1. **Molecular descriptors.** 2D Mordred descriptors are computed, and the 247 descriptors selected in the project are kept (`models/noms_colonnes_247_TC.txt`).
2. **Critical and thermo parameters (AI).** Ensemble neural networks predict the critical temperature *Tc*, the critical pressure *Pc*, the acentric factor *ω*, the normal boiling point *NBP*, the triple-point temperature *TTR* and the saturated-liquid molar volume at *Tr = 0.8*, *VL(Tr=0.8)*.
3. **Ideal-gas heat capacity (AI).** A neural network predicts *Cp°(T)/Cp,max* as a function of temperature. Multiplying by *Cp,max/R* gives *Cp°/R*.
4. **Cp°/R correlation.** The predicted *Cp°/R(T)* is fitted with a seven-coefficient correlation (A, B, C, D, E, F, G): A, B, D and F are dimensionless, C, E and G are in K.

   Cp°/R = A + B·(C/T)²·e^(C/T)/(e^(C/T) − 1)² + D·(E/T)²·e^(E/T)/(e^(E/T) − 1)² + F·(G/T)²·e^(G/T)/(e^(G/T) − 1)²

5. **tc-PR equation of state.** Peng-Robinson with a constant volume translation *c*, calibrated so that the EoS reproduces the predicted *VL(Tr=0.8)*:
   *c = v_liq,EoS(Tr = 0.8) − VL(Tr = 0.8)*. The calculation gives the stable phase and the molar density at a given *(T, P)*, the vapor pressure curve, the vapor-liquid coexistence, the enthalpy of vaporization and the heat capacity of the saturated liquid.

Steps 1-4 are available online (no installation needed):

- critical properties, *NBP*, *TTR* and *VL(Tr=0.8)*: <https://thermokinetics-lrgp.com/thermppt/>
- ideal-gas heat capacity and the seven coefficients: <https://thermokinetics-lrgp.com/cp-versus-temp/>

Step 5 is the script in `example/`, which takes the values given by these two tools as input.

## 2. Content of the folder

```
tcPR_Calcul/
├── README.md
├── models/
│   ├── noms_colonnes_247_TC.txt        list of the 247 Mordred descriptors
│   ├── 01_modele_final_TC.joblib       critical temperature        [K]
│   ├── 02_modele_final_PC.joblib       critical pressure           [bar]
│   ├── 03_modele_final_ACEN.joblib     acentric factor             [-]
│   ├── 04_modele_final_NBP.joblib      normal boiling point        [K]
│   ├── 05_modele_final_TTR.joblib      triple-point temperature    [K]
│   ├── 08_modele_final_VLIQ.joblib     liquid volume at Tr = 0.8   [cm3/mol]
│   └── 01_modele_final_Cp_CpMax.joblib ideal-gas Cp°(T)/Cp,max     [-]
└── example/
    ├── README.md
    └── tcPR_Calcul_Cp.py               stand-alone tc-PR calculation (1-propanol)
```

The models are ensembles of neural networks: the final prediction is the average of the sub-models. They are stored with a small helper class, `MetaModel`, which must be defined before loading the files (see below, and `train model/Class_Model.py` at the root of this repository).

## 3. Using the models in Python

Required packages: `numpy`, `pandas`, `scikit-learn`, `joblib`, `rdkit`, `mordredcommunity` (the maintained Mordred fork; it provides the `mordred` module). The files were tested with scikit-learn 1.8 and 1.9.

The snippet below, run from the `tcPR_Calcul` folder, predicts the parameters of 1-propanol and its ideal-gas *Cp°/R*:

```python
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.base import BaseEstimator, RegressorMixin


class MetaModel(BaseEstimator, RegressorMixin):
    """Average of the sub-models of an ensemble (needed to unpickle the .joblib files)."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return sum(m.predict(X) for m in self.models) / len(self.models)


smiles = "CCCO"                                    # 1-propanol
mol = Chem.MolFromSmiles(smiles)

# 247 selected Mordred descriptors
with open("models/noms_colonnes_247_TC.txt", encoding="utf-8-sig") as f:
    columns = [line.strip() for line in f if line.strip()]
if columns[0] == "0":                              # first line of the file is an index header
    columns = columns[1:]
desc = pd.DataFrame([Calculator(descriptors, ignore_3D=True)(mol).asdict()])
X = desc[columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

# Critical properties and thermo parameters
Tc   = joblib.load("models/01_modele_final_TC.joblib").predict(X)[0]     # K
Pc   = joblib.load("models/02_modele_final_PC.joblib").predict(X)[0]     # bar
acen = joblib.load("models/03_modele_final_ACEN.joblib").predict(X)[0]   # -
NBP  = joblib.load("models/04_modele_final_NBP.joblib").predict(X)[0]    # K
TTR  = joblib.load("models/05_modele_final_TTR.joblib").predict(X)[0]    # K
VL08 = joblib.load("models/08_modele_final_VLIQ.joblib").predict(X)[0]   # cm3/mol
print(f"Tc={Tc:.2f} K  Pc={Pc:.3f} bar  omega={acen:.4f}  NBP={NBP:.2f} K  "
      f"TTR={TTR:.2f} K  VL(Tr=0.8)={VL08:.3f} cm3/mol")

# Ideal-gas Cp/R at several temperatures
T = np.array([300., 500., 1000., 1500.])
N = Chem.AddHs(mol).GetNumAtoms()                  # total number of atoms, hydrogens included
cmax_over_R = 3 * N - 1.5 if N == 2 else 3 * N - 2
X_T = pd.concat([X] * len(T), ignore_index=True)
X_T["T(K)"] = T
cp_over_R = cmax_over_R * joblib.load("models/01_modele_final_Cp_CpMax.joblib").predict(X_T)
print(dict(zip(T, cp_over_R.round(4))))
```

Expected output for 1-propanol:

```
Tc=536.63 K  Pc=51.715 bar  omega=0.6211  NBP=370.86 K  TTR=148.52 K  VL(Tr=0.8)=92.988 cm3/mol
Cp/R at 300, 500, 1000, 1500 K: 10.2975, 15.4599, 23.2248, 26.9486
```

Notes:

- The models apply to molecules containing H, C, N, O, S, P, F, Cl, Br and I atoms (B for the heat capacity).
- *Cp,max/R* is `3N − 2` (`3N − 1.5` for a diatomic molecule), with *N* the total number of atoms including hydrogens.
- For methane and water, the online *thermppt* tool replaces some of the model outputs by fixed reference values (methane: Pc, NBP, ω, TTR; water: Tc, Pc).

## 4. Example

`example/` contains the stand-alone tc-PR calculation. See [`example/README.md`](example/README.md).

## 5. Related publications

1. R. Bounaceur, F. Paes, R. Privat, J.-N. Jaubert, *AI-powered prediction of critical properties and boiling points: a hybrid ensemble learning and QSPR approach*, Journal of Cheminformatics (2025). <https://doi.org/10.1186/s13321-025-01062-9> (open access)
2. R. Bounaceur, F. Paes, J.-N. Jaubert, R. Privat, *Accurate prediction of ideal gas heat capacity Cp(T) using machine learning: data augmentation and ensemble learning for robust thermodynamic modeling*, Chemical Engineering Journal Advances (2026). <https://doi.org/10.1016/j.ceja.2026.101140> (open access)
3. F. Paes, R. Privat, R. Bounaceur, J.-N. Jaubert, *Prediction of Pure-Component Thermodynamic Properties Using the Peng–Robinson Equation of State with Machine Learning-Derived Input Parameters*, Industrial & Engineering Chemistry Research (2026). <https://doi.org/10.1021/acs.iecr.6c00300>
4. A. Piña-Martinez, R. Privat, J.-N. Jaubert, *Use of 300,000 pseudo-experimental data over 1800 pure fluids to assess the performance of four cubic equations of state: SRK, PR, tc-RK, and tc-PR*, AIChE Journal 68(2) (2021). <https://doi.org/10.1002/aic.17518> (open access)

## 6. Authors and contact

Roda Bounaceur, Francisco Paes, Romain Privat, Jean-Noël Jaubert
Université de Lorraine, CNRS, LRGP, F-54000 Nancy, France

Contact: roda.bounaceur@univ-lorraine.fr

## 7. License

GNU General Public License v3.0 (see the `LICENSE` file at the root of the repository).
