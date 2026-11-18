# -*- coding: utf-8 -*-
"""
Calcul des propriétés de l'équation de Peng-Robinson, version tc-PR
RP
"""

# Bibliothèques Python : 
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import numpy.polynomial.polynomial as nppol # résolution équation polynomiale

# Pour les prédictions IA à partir du SMILES :
from pathlib import Path
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from sklearn.base import BaseEstimator, RegressorMixin


class MetaModel(BaseEstimator, RegressorMixin):
    # Classe utilisée pour sauvegarder les modèles ensemble du LRGP
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        return sum(model.predict(X) for model in self.models) / len(self.models)


# Permet de relire les modèles sauvegardés avec Class_Model.MetaModel :
sys.modules.setdefault('Class_Model', sys.modules[__name__])

nfig = 0 # compteur de figures

# Seule donnée d'entrée à renseigner :
SMILES = 'CCCCCCO'

# Modèles IA et liste des 247 descripteurs placés dans le même répertoire :
repertoire = Path(__file__).resolve().parent
mol = Chem.MolFromSmiles(SMILES)
if mol is None:
    raise ValueError('SMILES invalide : ' + SMILES)
if len(Chem.GetMolFrags(mol)) != 1:
    raise ValueError('Le SMILES doit décrire une seule molécule.')

calc = Calculator(descriptors, ignore_3D=False)
df = calc.pandas([mol], nproc=1) # sans multiprocessing sous Spyder

with open(repertoire / 'noms_colonnes_247_TC.txt', encoding='utf-8-sig') as f:
    noms_colonnes_retenues = [ligne.strip() for ligne in f if ligne.strip()]
if noms_colonnes_retenues[0] == '0':
    del noms_colonnes_retenues[0]
if len(noms_colonnes_retenues) != 247:
    raise ValueError('La liste doit comporter 247 descripteurs Mordred.')

colonnes_absentes = [nom for nom in noms_colonnes_retenues if nom not in df.columns]
if colonnes_absentes:
    raise ValueError('Descripteurs Mordred absents : ' + ', '.join(colonnes_absentes))

X = df[noms_colonnes_retenues].copy()
colonnes_non_numeriques = []
for col in X.columns:
    valeurs = pd.to_numeric(X[col], errors='coerce')
    if valeurs.isna().any():
        colonnes_non_numeriques.append(col)
    X[col] = valeurs.fillna(0)
X = X.fillna(0).astype(float)
if colonnes_non_numeriques:
    warnings.warn(str(len(colonnes_non_numeriques)) +
                  ' descripteurs manquants/non numériques, remplacés par 0 : ' +
                  ', '.join(colonnes_non_numeriques[:12]))


def verifier_entrees(modele, donnees, nom):
    # Contrôle des variables attendues, si les modèles contiennent ces métadonnées.
    modeles = [modele] + list(getattr(modele, 'models', []))
    for m in modeles:
        if hasattr(m, 'n_features_in_') and m.n_features_in_ != donnees.shape[1]:
            raise ValueError(nom + ' : nombre de descripteurs IA incompatible.')
        if hasattr(m, 'feature_names_in_') and list(m.feature_names_in_) != list(donnees.columns):
            raise ValueError(nom + ' : ordre ou noms des descripteurs IA incompatibles.')

# Coordonnées critiques prédites par IA :
TC = joblib.load(repertoire / '01_modele_final_TC.joblib')
PC = joblib.load(repertoire / '02_modele_final_PC.joblib')
ACEN = joblib.load(repertoire / '03_modele_final_ACEN.joblib')
modeleVLIQ = joblib.load(repertoire / '08_modele_final_VLIQ.joblib')
modeleCp = joblib.load(repertoire / '01_modele_final_Cp_CpMax.joblib')

verifier_entrees(TC, X, 'Tc')
verifier_entrees(PC, X, 'Pc')
verifier_entrees(ACEN, X, 'omega')
verifier_entrees(modeleVLIQ, X, 'Vliq at Tr=0.8')

Tc = float(TC.predict(X)[0]) # température critique, K
Pc = float(PC.predict(X)[0])*1e5 # pression critique, Pa (sortie IA en bar)
w = float(ACEN.predict(X)[0]) # facteur acentrique
if Tc <= 0 or Pc <= 0 or not np.all(np.isfinite([Tc, Pc, w])):
    raise ValueError('Paramètres critiques IA non physiques.')
if not (30. <= Tc <= 2000.):
    warnings.warn('Tc hors de la plage de contrôle 30–2000 K : vérifier la prédiction IA.')
if not (0.1 <= Pc*1e-5 <= 500.):
    warnings.warn('Pc hors de la plage de contrôle 0.1–500 bar : vérifier les unités IA.')
if not (-0.5 <= w <= 2.):
    warnings.warn('Facteur acentrique inhabituel : vérifier la prédiction IA.')
print('SMILES =', SMILES, '| Tc (K) =', Tc, '| Pc (bar) =', Pc*1e-5, '| w =', w)

# Volume molaire liquide à Tr=0.8 prédit directement par IA :
# La valeur brute est convertie en m3/mol après le calcul de vcrit.
Rgaz = 8.31446261815324 # Constante des gaz parfaits, J/mol/K
vliq08_brut = float(modeleVLIQ.predict(X)[0])
if not np.isfinite(vliq08_brut) or vliq08_brut <= 0.:
    raise ValueError('Prédiction IA de Vliq à Tr=0.8 non physique.')

# Cp du gaz parfait :
# Modèle IA : la sortie est le rapport Cp(T) / Cpmax.
molH = Chem.AddHs(mol)
nbAt_Tot = molH.GetNumAtoms()

# La géométrie concerne tous les atomes, hydrogènes compris.
if nbAt_Tot == 1:
    geometrie = 'Monatomic'
    CMax = 2.5*Rgaz
elif nbAt_Tot == 2:
    geometrie = 'Linear'
    CMax = (3.0*nbAt_Tot - 1.5)*Rgaz
elif any(atome.GetDegree() >= 3 for atome in molH.GetAtoms()):
    geometrie = 'Non-linear'
    CMax = (3.0*nbAt_Tot - 2.0)*Rgaz
else:
    mol3D = Chem.Mol(molH)
    parametres3D = AllChem.ETKDGv3()
    parametres3D.randomSeed = 2026
    if AllChem.EmbedMolecule(mol3D, parametres3D) < 0:
        raise RuntimeError('Impossible de déterminer la géométrie 3D de ' + SMILES)
    coordonnees = np.array(mol3D.GetConformer().GetPositions())
    coordonnees -= coordonnees.mean(axis=0)
    valeurs = np.linalg.svd(coordonnees, compute_uv=False)
    geometrie = 'Linear' if valeurs[1]/valeurs[0] < 0.03 else 'Non-linear'
    CMax = (3.0*nbAt_Tot - (1.5 if geometrie == 'Linear' else 2.0))*Rgaz

X_Cp_test = X.copy()
X_Cp_test['T(K)'] = 0.8*Tc
verifier_entrees(modeleCp, X_Cp_test[noms_colonnes_retenues + ['T(K)']], 'Cp ideal gas')

print('Molecular geometry =', geometrie, '| N =', nbAt_Tot,
      '| Cp,max/R =', CMax/Rgaz, '| Cp,max (J/mol/K) =', CMax)

# Autres constantes :
onethird = 1./3. # puissance 1/3
eps = 1e-9 # tolérance relative sur la pression de saturation
itmax = int(1e5)
methodes_Psat = {'derivative': 0, 'fallback': 0}

# Constantes de l'équation de Van der Waals : 
r1 = -1. - np.sqrt(2.)
r2 = -1. + np.sqrt(2.)
sumr = r1 + r2

compac_crit = 1./(((1-r1)*(1-r2)**2)**onethird \
                  + ((1-r2)*(1-r1)**2)**onethird + 1.) # compacité critique = b/Vcrit
zcrit = 1. / (3.-compac_crit*(1+sumr)) # facteur de compressibilité critique
omega_b = compac_crit * zcrit
omega_a = (1-compac_crit*r1)*(1-compac_crit*r2)*(2-compac_crit*sumr) \
        /((1-compac_crit)*(3-compac_crit*(1+sumr))**2)
RTc = Rgaz*Tc
acrit = omega_a*RTc**2 / Pc # a(Tc)
b = omega_b * RTc / Pc # covolume, m3/mol
vcrit = zcrit * RTc / Pc # Volume molaire critique, m3/mol

# Vérification de l'unité du modèle VLIQ : m3/mol, L/mol ou cm3/mol.
# Une seule interprétation doit donner un volume liquide plausible à Tr=0.8.
conversions_VLIQ = {'m3/mol': 1., 'L/mol': 1e-3, 'cm3/mol': 1e-6}
volumes_possibles = [(unite, vliq08_brut*facteur)
                    for unite, facteur in conversions_VLIQ.items()
                    if 0.05*vcrit < vliq08_brut*facteur < vcrit]
if len(volumes_possibles) != 1:
    raise ValueError('Unité ou valeur prédite ambiguë pour Vliq à Tr=0.8 : ' +
                     str(vliq08_brut) +
                     '. Vérifier l’unité de sortie du fichier 08_modele_final_VLIQ.joblib.')
unite_VLIQ, vliq08 = volumes_possibles[0]
print('AI liquid volume at Tr=0.8 =', vliq08_brut, unite_VLIQ,
      '| Vliq (m3/mol) =', vliq08)

# Fonction a(T) : 
#==============================================================================
def m(w): # facteur de forme
    return 0.37464 + 1.54226*w - 0.26992*w**2 # m de PR

def Coeff_a_EoS(T,Tc,w,ider): # paramètre attractif a, Pa.m^6 / mol²
    # Inputs = temp. T (K), temp. crit. Tc (K), fact. acent. w, ider
    # ider : niveau de dérivation souhaité
    #    ider = 0, a
    #    ider = 1, a'(T)
    #    ider = 2, a''(T)
    fact_m = m(w)
    foncT = 1+fact_m*(1. - np.sqrt(T/Tc))
    a_EoS = acrit*foncT**2
    da_EoS = 0. 
    d2a_EoS = 0. 
    
    if ider >= 1:
        da_EoS = -acrit*fact_m*foncT / np.sqrt(T*Tc) # da / dT
    if ider >= 2:
        d2a_EoS = 0.5*acrit*fact_m*(fact_m+1)/np.sqrt(Tc)*T**(-1.5) # d²a / dT²
    return a_EoS, da_EoS, d2a_EoS

# Fonction P(T,v,a,b,c), pression en Pa
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def P_EoS(T,v,a,b,c): # Pa
    v_nt = v + c # volume non translaté
    return Rgaz*T/(v_nt-b) - a/(v_nt-r1*b)/(v_nt-r2*b) 

# log nat. de la fugacité
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def lnfug_EoS(T,v,a,b,c):
    RT = Rgaz*T
    pres = P_EoS(T,v,a,b,c)
    v_nt = v + c # volume non translaté
    return pres*v_nt/RT - 1. - np.log((v_nt-b)/RT) \
        + a/(RT*b*(r1-r2))*np.log((v_nt-b*r1)/(v_nt-b*r2)) \

# Fonction hres_EoS(T,v,a,b,c), pression en Pa
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def hres_EoS(T,v,a,b,c,da): # Pa
    RT = Rgaz*T
    pres = P_EoS(T,v,a,b,c)
    v_nt = v + c # volume non translaté
    vbr1 = v_nt-b*r1
    vbr2 = v_nt-b*r2
    return RT*b/(v_nt-b) - a*v_nt/(vbr1*vbr2) \
        + 1/(b*(r1-r2))*(a - T*da)*np.log(vbr1/vbr2) \

# Fonction cPres_EoS(T,v,a,b,c), pression en Pa
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def Cpres_EoS(T,v,a,b,c,da,d2a): # Pa
    RT = Rgaz*T
    v_nt = v + c # volume non translaté
    vb = v_nt - b
    vbr1 = v_nt-b*r1
    vbr2 = v_nt-b*r2
    Cvres = -T/(b*(r1-r2))*d2a*np.log(vbr1/vbr2) 
    kTv = 1/(RT/vb**2 - a*(2*v_nt- (r1+r2)*b)/(vbr1*vbr2)**2)
    betaP = Rgaz/(v_nt-b) - da/(vbr1*vbr2)
    return Cvres - Rgaz + T * kTv * betaP**2

# Fonction CpGP(T), chaleur spécifique du gaz parfait prédite par IA
#==============================================================================
def CpGP(T): # J/mol/K
    Xp = X.copy()
    Xp['T(K)'] = T
    Cp_ratio = modeleCp.predict(Xp[noms_colonnes_retenues + ['T(K)']])
    return float(np.asarray(Cp_ratio).reshape(-1)[0])*CMax # J/mol/K

# Résolution de l'équation d'état à T et P fixées : 
# il s'agit de déterminer les racines v de l'EoS qui sont solutions
# d'une équation cubique polynomiale de degré 3.
# Inputs : T, temperature (K) et P, pression en Pa
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
# Output : vecteur des volumes molaires solutions de l'EoS
#==============================================================================

# -------------------
# Aparté PYTHON : résolution d'une équation polynomiale à partir de polyroots.
# Exemple : pour calculer les racines du polynôme x^2 + 3.x + 5 = 0, on écrit : 
#           X = nppol.polyroots([5, 3, 1])
# -------------------

# écriture généralisée d'une cubique : P = RT/(v-b) - a / [(v-b.r1)(v-b.r2)]
def resolEoS(T,P,a,b,c): # résolution d'une équation cubique à T (K) et P (Pa) fixées
    # Méthode : on résout l'EoS non translatée puis on corrige les volumes
    # obtenus en soustrayant c
    RT = Rgaz*T 
    coef3 = 1.
    coef2 = -(b*(r1 + r2 + 1) + RT/P)
    coef1 = b**2 * (r1*r2 + r1 + r2) + RT*b/P*(r1+r2) + a/P
    coef0 = -b*(r1*r2*b**2 + r1*r2*b*RT/P + a/P)
    volumes = nppol.polyroots([coef0, coef1, coef2, coef3])
    
    # Sélection des racines réelles :
    volumes = volumes.real[abs(volumes.imag) < 1e-7*b]

    # suppression des racines non physiques telles que v < b:
    volumes = volumes[volumes >= b] - c
    return volumes

# Extrema de l'isotherme obtenus par (dP/dv)_T = 0 :
# résolution d'un polynôme de degré 4 en x = (v+c)/b.
#==============================================================================
def Bornes_Psat_derivee(T,a,c):
    A = a/(Rgaz*T*b)
    pol1 = nppol.polymul([-sumr,2.], nppol.polypow([-1.,1.],2))
    pol2 = nppol.polymul(nppol.polypow([-r1,1.],2),
                        nppol.polypow([-r2,1.],2))
    racines = nppol.polyroots(nppol.polysub(A*pol1,pol2))
    x = np.sort(racines.real[(abs(racines.imag)<1e-7) & (racines.real>1.)])
    if len(x) != 2 or x[1]-x[0] <= 1e-8:
        raise RuntimeError('Extrema de la dérivée non identifiés.')
    Pmin = P_EoS(T,x[0]*b-c,a,b,c)
    Pmax = P_EoS(T,x[1]*b-c,a,b,c)
    if not np.all(np.isfinite([Pmin,Pmax])) or Pmax <= max(Pmin,0.):
        raise RuntimeError('Bornes de pression non physiques.')
    return max(Pmin,0.), Pmax

# Procédure initiale de recherche des extrema : balayage volumique.
# Conservée comme procédure de secours.
#==============================================================================
def Bornes_Psat_balayage(T,a,c):
    bnew = b - c # covolume corrigé
    stepv = 1e-2*bnew

    # Recherche de Pmin :
    vnew = bnew
    Pold = 1e300
    dP = -1.
    it = 0
    while dP<0:
        vold = vnew
        vnew = vold + stepv
        Pnew = P_EoS(T,vnew,a,b,c)
        dP = Pnew - Pold
        Pold = Pnew
        it += 1
        if it > itmax:
            raise RuntimeError('Balayage Pmin : trop d’itérations.')

    Pmin = max(Pold,0.)

    # Recherche de Pmax :
    it = 0
    while dP>0:
        vold = vnew
        vnew = vold + stepv
        Pnew = P_EoS(T,vnew,a,b,c)
        dP = Pnew - Pold
        Pold = Pnew
        it += 1
        if it > itmax:
            raise RuntimeError('Balayage Pmax : trop d’itérations.')

    Pmax = Pold
    if not np.all(np.isfinite([Pmin,Pmax])) or Pmax <= Pmin:
        raise RuntimeError('Balayage : bornes de pression invalides.')
    return Pmin, Pmax

# Dichotomie avec tolérance relative sur la pression et contrôle des phases.
#==============================================================================
def Dichotomie_Psat(T,a,c,Pmin,Pmax):
    if Pmax <= Pmin or Pmax <= 0.:
        raise RuntimeError('Intervalle de saturation invalide.')
    for i in range(100):
        P0 = 0.5*(Pmin + Pmax)
        if P0 <= 0.:
            raise RuntimeError('Pression de saturation non positive.')
        VolMol = resolEoS(T,P0,a,b,c)
        if len(VolMol) < 2:
            raise RuntimeError('Deux phases distinctes introuvables.')
        volLiq = min(VolMol)
        volGaz = max(VolMol)
        if volLiq <= 0. or volGaz <= volLiq*(1.+1e-8):
            raise RuntimeError('Volumes liquide/vapeur non physiques ou triviaux.')

        delta = lnfug_EoS(T,volLiq,a,b,c) - lnfug_EoS(T,volGaz,a,b,c)
        if not np.isfinite(delta):
            raise RuntimeError('Fugacités non finies.')
        if delta > 0.:
            Pmin = P0
        else:
            Pmax = P0
        if Pmax-Pmin <= eps*max(P0,1.):
            return P0, volLiq, volGaz
    raise RuntimeError('Dichotomie de saturation non convergée.')

# Calcul d'une pression de vapeur saturante à T fixée :
#==============================================================================
def Psat(T,c):
    # Psat en Pa ; T en K ; c en m3/mol ; volumes en m3/mol.
    if T>=Tc:
        return 0., 0., 0.

    a, da, d2a = Coeff_a_EoS(T,Tc,w,0)
    erreur_derivee = None
    for methode, recherche in [('derivative',Bornes_Psat_derivee),
                               ('fallback',Bornes_Psat_balayage)]:
        try:
            Pmin, Pmax = recherche(T,a,c)
            resultat = Dichotomie_Psat(T,a,c,Pmin,Pmax)
            methodes_Psat[methode] += 1
            return resultat
        except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as erreur:
            if methode == 'derivative':
                erreur_derivee = erreur
            else:
                raise RuntimeError('ELV non convergé à T = ' + str(T) +
                                   ' K (dérivée : ' + str(erreur_derivee) +
                                   ' ; balayage : ' + str(erreur) + ')') from erreur

# Calcul du paramètre de translation volumique c de l'EoS : 
#==============================================================================
def Calcul_c():
    # Outputs : - cvol, correction volumique en m3/mol
    #             Elle est telle que : vEoS = vEoS non corrigée - cvol
    #           - bnew, m3/mol, nouveau covolume = b EoS non corrigée - cvol
    c = 0
    T08 = 0.8*Tc
    # Calcul du vol mol liq saturé à Tr = 0.8
    bidon, veos_liq08, bidon = Psat(T08,c)
    # Calcul de la translation vol :
    # c = Vliq EoS originale (Tr = 0.8) - Vliq IA (Tr = 0.8)
    cvol = veos_liq08 - vliq08
    bnew = b - cvol
    if bnew <= 0:
        raise ValueError('Translation volumique non physique (b-c <= 0).')
    return cvol, bnew

#==============================================================================
#==============================================================================
# MAIN :
#
# Calcul de la correction volumique c :
#------------------------------------------------------------------------------
c, bnew = Calcul_c()
vcrit = vcrit - c # volume molaire critique après translation
print('Volume translation c (m3/mol) =', c)

# Calcul et tracé de Psat(T)
#------------------------------------------------------------------------------
T = 0.9*Tc # K, vérification de la pression de saturation
try:
    print("Psat (Pa), vLiq (m3/mol), vGaz (m3/mol) = ", Psat(T,c))
except RuntimeError as erreur:
    warnings.warn(str(erreur))

Tmin = 0.7*Tc # K
Tmax = np.floor(Tc) 
DT = 0.5
T_List = np.arange(Tmin, Tmax, DT)
Psat_List = np.zeros(len(T_List))
Vliq_List = np.zeros(len(T_List))
Vgaz_List = np.zeros(len(T_List))
DvapH_List = np.zeros(len(T_List))

CpLiq_List = []
CpGP_List = []
T_ListCp = []
anomalies = []

for i in range(len(T_List)):
    T = T_List[i]
    try:
        res = Psat(T,c)
    except RuntimeError as erreur:
        anomalies.append(str(erreur))
        Psat_List[i] = np.nan
        Vliq_List[i] = np.nan
        Vgaz_List[i] = np.nan
        DvapH_List[i] = np.nan
        continue
    vl = res[1]
    vg = res[2]
    
    Psat_List[i] = res[0]*1e-5  # bar
    Vliq_List[i] = res[1]*1e6 # mL/mol
    Vgaz_List[i] = res[2]*1e6 # mL/mol
    if (not np.all(np.isfinite(res)) or res[0] <= 0 or res[0] >= Pc
            or vl <= 0 or vg <= vl):
        anomalies.append('Phase equilibrium at T = ' + str(T) + ' K')
    
    # Calcul du param attractif a à la température T :
    a, da, d2a = Coeff_a_EoS(T,Tc,w,2)

    DvapH_List[i] = hres_EoS(T,vg,a,b,c,da) \
        - hres_EoS(T,vl,a,b,c,da)
    if T < 0.98*Tc:
        T_ListCp = np.append(T_ListCp,T)
        Cp0 = CpGP(T)
        CpGP_List = np.append(CpGP_List, Cp0)
        CpL = Cpres_EoS(T,vl,a,b,c,da,d2a) + Cp0
        CpLiq_List = np.append(CpLiq_List, CpL)
        if not np.all(np.isfinite([Cp0, CpL])) or Cp0 <= 0 or CpL <= 0 or Cp0 > CMax:
            anomalies.append('Heat capacity at T = ' + str(T) + ' K')

print('Psat solver: derivative =', methodes_Psat['derivative'],
      '| original volume-scan fallback =', methodes_Psat['fallback'])
if anomalies:
    warnings.warn(str(len(anomalies)) + ' anomalous calculated point(s): ' +
                  '; '.join(anomalies[:5]))
else:
    print('Physical checks: no anomaly detected on the calculated curves.')

# Point critique : 
T_List = np.append(T_List,Tc)
Psat_List = np.append(Psat_List,Pc*1e-5) # bar
Vliq_List = np.append(Vliq_List,vcrit*1e6) # mL/mol
Vgaz_List = np.append(Vgaz_List,vcrit*1e6) # mL/mol
DvapH_List = np.append(DvapH_List,0.)

# T = 289.517
# a, da, d2a = Coeff_a_EoS(T,Tc,w,2)
# print("T",T)
# Ps,vl,vg = Psat(T,c)
# print('Cpres EoS = ',Cpres_EoS(T,vl,a,b,c,da,d2a))
# print('CpGP EoS = ',CpGP(T))
# sys.exit()

#************ TRACE DU PLAN (P, T)
nfig += 1
plt.figure(nfig)
#************
plt.clf()
plt.plot(T_List, Psat_List,'k')
plt.plot(Tc, Pc*1e-5,'ko')
plt.xlabel('$T$ (K)',fontsize=10)
plt.ylabel('$P^{sat}$ (bar)',fontsize=10)
plt.title("Vapor-pressure curve\nSMILES: " + SMILES,fontsize=18)
plt.show()

#************ TRACE DU PLAN (T, densité)
nfig += 1
plt.figure(nfig)
#************
vcrit_cm3 = vcrit*1e6 # cm3/mol
vliq08_cm3 = vliq08*1e6
plt.clf()
plt.plot(1e3/Vliq_List , T_List,'b',label="Saturated liquid")
plt.plot(1e3/Vgaz_List , T_List,'r',label="Saturated vapor")
plt.plot(1e3/vcrit_cm3, Tc,'ko',label = "Critical point")
plt.plot(1e3/vliq08_cm3, 0.8*Tc,'g*',label = r"AI liquid volume at $T_r=0.8$")
plt.xlabel(r'$\rho$ (mol/L)',fontsize=10)
plt.ylabel('$T$ (K)',fontsize=10)
plt.title("Saturation envelope\nSMILES: " + SMILES,fontsize=18)
plt.legend(fontsize=10)
plt.show()

#************ TRACE DU PLAN (T, enthalpie de vaporisation)
nfig += 1
plt.figure(nfig)
#************
plt.clf()
plt.plot(T_List , DvapH_List*1e-3,'b')
plt.xlabel(r'$T$ (K)',fontsize=10)
plt.ylabel(r'$\Delta_{vap}H$ (kJ/mol)',fontsize=10)
plt.title("Enthalpy of vaporization\nSMILES: " + SMILES,fontsize=18)
plt.show()


#************ TRACE DU PLAN (T, cP liq)
nfig += 1
plt.figure(nfig)
#************
vcrit_cm3 = vcrit*1e6 # cm3/mol
vliq08_cm3 = vliq08*1e6
plt.clf()
plt.plot(T_ListCp, CpLiq_List,'b',label="Saturated liquid")
plt.xlabel(r'$T$ (K)',fontsize=10)
plt.ylabel(r'$c_{P,liq}$ (J/mol/K)',fontsize=10)
plt.title("Saturated-liquid heat capacity\nSMILES: " + SMILES,fontsize=18)
plt.legend(fontsize=10)
plt.show()

#************ IDEAL-GAS HEAT CAPACITY
nfig += 1
plt.figure(nfig)
#************
plt.clf()
plt.plot(T_ListCp, CpGP_List,'b',label="AI prediction")
plt.xlabel(r'$T$ (K)',fontsize=10)
plt.ylabel(r'$C_{P}^{\circ}$ (J/mol/K)',fontsize=10)
plt.title("Ideal-gas heat capacity\nSMILES: " + SMILES,fontsize=18)
plt.legend(fontsize=10)
plt.show()
