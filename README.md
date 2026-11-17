# AI-ThermPpt

AI-powered prediction of critical properties and boiling points: a hybrid ensemble learning and QSPR approach, and its use in the volume-translated Peng-Robinson equation of state (tc-PR).

This repository gathers the codes, data and trained models developed at the LRGP (Université de Lorraine, CNRS, Nancy) for the following publications:

1. R. Bounaceur, F. Paes, R. Privat, J.-N. Jaubert, *AI-powered prediction of critical properties and boiling points: a hybrid ensemble learning and QSPR approach*, Journal of Cheminformatics (2025). <https://doi.org/10.1186/s13321-025-01062-9> (open access)
2. R. Bounaceur, F. Paes, J.-N. Jaubert, R. Privat, *Accurate prediction of ideal gas heat capacity Cp(T) using machine learning: data augmentation and ensemble learning for robust thermodynamic modeling*, Chemical Engineering Journal Advances (2026). <https://doi.org/10.1016/j.ceja.2026.101140> (open access)
3. F. Paes, R. Privat, R. Bounaceur, J.-N. Jaubert, *Prediction of Pure-Component Thermodynamic Properties Using the Peng–Robinson Equation of State with Machine Learning-Derived Input Parameters*, Industrial & Engineering Chemistry Research (2026). <https://doi.org/10.1021/acs.iecr.6c00300>
4. A. Piña-Martinez, R. Privat, J.-N. Jaubert, *Use of 300,000 pseudo-experimental data over 1800 pure fluids to assess the performance of four cubic equations of state: SRK, PR, tc-RK, and tc-PR*, AIChE Journal 68(2) (2021). <https://doi.org/10.1002/aic.17518> (open access)

## How the project is built

Starting from a SMILES notation only, the project predicts the parameters needed by a thermodynamic model:

1. **Molecular descriptors.** The Mordred calculator gives the 2D descriptors of the molecule. After a statistical analysis of the complete database, 247 descriptors are retained (`noms_colonnes_247_TC.txt`).
2. **Critical properties and boiling point (publication 1).** For each property (critical temperature *Tc*, critical pressure *Pc*, acentric factor *ω*, normal boiling point *NBP*), several neural networks are trained in a bagging framework, and the final model is the average of these sub-models. This is the "ensemble learning" approach: the helper class `MetaModel` performs the averaging. The models cover molecules containing C, H, O, N, S, P, F, Cl, Br and I atoms.
3. **Ideal-gas heat capacity (publication 2).** A neural network, trained on an experimental database augmented with validated correlations, predicts *Cp°(T)* as a continuous function of temperature.
4. **Equation of state (publications 3 and 4).** The predicted parameters are the inputs of the volume-translated Peng-Robinson equation of state (tc-PR), which gives vapor pressure, densities, enthalpy of vaporization and heat capacities.

Online tools using these models (no installation needed):

- critical properties, *NBP*, triple-point temperature and liquid volume at *Tr = 0.8*: <https://thermokinetics-lrgp.com/thermppt/>
- ideal-gas heat capacity: <https://thermokinetics-lrgp.com/cp-versus-temp/>

## Content of the repository

```
AI-ThermPpt/
├── data/                  training database and comparison with the ICAS method
├── train model/           final trained models (Tc, Pc, acentric factor, NBP) and MetaModel class
├── example/               worked example of the critical-property models (alkanes)
├── yaml_files/            Anaconda environment to reproduce the calculations
├── tcPR_Calcul/           AI-derived parameters (6 properties + Cp°) used in the tc-PR equation of state
├── requirements.txt
├── LICENSE
└── README.md
```

### `data/`

- `DataBase_LRGP.csv`: the 1705 compounds of the database (identifier, name, formula, CAS number, SMILES, chemical family, molar mass).
- `ICAS_LRGP_Comparison.csv`: critical temperature, critical pressure and acentric factor obtained with the ICAS method, compared with the LRGP models.

### `train model/`

- `01_modele_final_TC.joblib`, `02_modele_final_PC.joblib`, `03_modele_final_ACEN.joblib`, `04_modele_final_NBP.joblib`: the final models, trained on the whole database (outputs: *Tc* in K, *Pc* in bar, *ω*, *NBP* in K).
- `Class_Model.py`: the `MetaModel` class, required to load the models (ensemble average of the sub-models).
- `noms_colonnes_247_TC.txt`: the list of the 247 selected descriptors.

### `example/`

A complete example, `Fichier_Exemple.py` (and the notebook `Fichier_Exemple.ipynb`):

1. define the `MetaModel` class so that the AI models can be interpreted;
2. load the models (`.joblib` files);
3. read the list of the selected descriptors;
4. compute the complete list of Mordred descriptors of each molecule (here the alkanes of `Liste_Alcanes.txt`);
5. after a quick clean-up of the values, run the models to obtain the thermodynamic parameters.

### `yaml_files/`

`Environnement_Mordred_Roda.yaml` is the Anaconda environment used to reproduce the calculations. It can be used directly with `conda env create -f Environnement_Mordred_Roda.yaml`.

### `tcPR_Calcul/`

Extension of the project to the tc-PR equation of state. It contains:

- `models/`: the models for the critical temperature, critical pressure, acentric factor, normal boiling point, triple-point temperature, saturated-liquid volume at *Tr = 0.8* and ideal-gas *Cp°(T)*, with the descriptor list;
- `example/`: a stand-alone script that computes, with tc-PR, the stable phase and density at a given *(T, P)*, the vapor-pressure curve, the vapor-liquid coexistence, the enthalpy of vaporization and the saturated-liquid heat capacity of a compound (1-propanol as worked example).

See [`tcPR_Calcul/README.md`](tcPR_Calcul/README.md) for the details.

## Installation

To reproduce the calculations with the original environment, use the Anaconda file of `yaml_files/`. Otherwise, install the necessary packages with:

```bash
pip install -r requirements.txt
```

For the models and the examples, the following packages are used: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`, `rdkit` and `mordredcommunity` (the maintained fork of Mordred, which provides the `mordred` module). The stand-alone tc-PR example needs only `numpy` and `matplotlib`.

## Authors and contact

Roda Bounaceur, Francisco Paes, Romain Privat, Jean-Noël Jaubert
Université de Lorraine, CNRS, LRGP, F-54000 Nancy, France

Contact: roda.bounaceur@univ-lorraine.fr

## License

GNU General Public License v3.0, see the `LICENSE` file.
