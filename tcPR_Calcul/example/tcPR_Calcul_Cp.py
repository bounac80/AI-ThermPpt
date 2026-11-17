# -*- coding: utf-8 -*-
"""
Calcul des propriétés de l'équation de Peng-Robinson, version tc-PR
RP
"""

# Bibliothèques Python : 
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as nppol # résolution équation polynomiale
from pathlib import Path
from datetime import datetime

nfig = 0 # compteur de figures

# =============================================================================
# DONNÉES À RENSEIGNER : exemple du 1-propanol (CCCO)
# =============================================================================
# État thermodynamique à étudier :
T = 298.15  # température, K
P = 1.0e5   # pression, Pa

# Coordonnées critiques et facteur acentrique :
# Si inconnus : https://thermokinetics-lrgp.com/thermppt/
Tc = 536.63       # température critique, K
Pc = 51.71e5      # pression critique, Pa (51.71 bar)
w = 0.6211        # facteur acentrique

# Donnée de densité liquide :
# Si inconnue : https://thermokinetics-lrgp.com/thermppt/
vliq08 = 92.99e-6  # m3/mol, volume mol. liq à Tr = 0.8

# Cp du gaz parfait : coefficients de la corrélation DIPPR 127 
# il sont fournis par le site
# https://thermokinetics-lrgp.com/cp-versus-temp/
# ATTENTION : les coefficients affichés sur ce site décrivent Cp°/R
# (coefficients A, B, D, F sans dimension ; C, E, G en K).
CoeffCpGP = [3.3236, 3.0552, 349.63, 13.677, 1193.3, 11.669, 3460.0]

# Autres constantes :
Rgaz = 8.31446261815324 # Constante des gaz parfaits, J/mol/K
onethird = 1./3. # puissance 1/3
eps = 1e-9 # tolérance relative sur la pression de saturation
itmax = int(1e5)

if T <= 0 or P <= 0 or Tc <= 0 or Pc <= 0 or vliq08 <= 0:
    raise ValueError('T, P, Tc, Pc et vliq08 doivent être strictement positifs.')
if len(CoeffCpGP) != 7 or not np.all(np.isfinite(CoeffCpGP)):
    raise ValueError('Il faut sept coefficients DIPPR 127 finis.')
if not np.all(np.isfinite([T,P,Tc,Pc,w,vliq08])):
    raise ValueError('Les données thermodynamiques doivent être finies.')

methodes_Psat = {'dérivée': 0, 'balayage de secours': 0}

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

# Fonction cPGP(T), pression en Pa
# Equation 127 de la DIPPR
#==============================================================================
def CpGP(CoeffCpGP,T): # J/mol/K
    A, B, C, D, E, F, G = CoeffCpGP
    term1 = A
    term2 = (B * C**2 / T**2) * (np.exp(C / T) / (np.exp(C / T) - 1)**2)
    term3 = (D * E**2 / T**2) * (np.exp(E / T) / (np.exp(E / T) - 1)**2)
    term4 = (F * G**2 / T**2) * (np.exp(G / T) / (np.exp(G / T) - 1)**2)
    return Rgaz*(term1 + term2 + term3 + term4) # J/mol/K (coefficients Cp°/R du site)

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
    volumes = volumes.real[abs(volumes.imag)<1e-7*b]

    # suppression des racines non physiques telles que v < b:
    volumes = volumes[volumes >= b] - c
    return np.sort(volumes[volumes > 0.])

# Extrema de l'isotherme : (dP/dv)_T = 0
# Résolution d'un polynôme du 4e degré en x = (v+c)/b.
#==============================================================================
def Bornes_Psat_derivee(T,a,c):
    A = a/(Rgaz*T*b)
    pol1 = nppol.polymul([-sumr,2.],nppol.polypow([-1.,1.],2))
    pol2 = nppol.polymul(nppol.polypow([-r1,1.],2),
                        nppol.polypow([-r2,1.],2))
    racines = nppol.polyroots(nppol.polysub(A*pol1,pol2))
    x = np.sort(racines.real[(abs(racines.imag)<1e-7) & (racines.real>1.)])
    if len(x) != 2 or x[1]-x[0] <= 1e-8:
        raise RuntimeError('Les extrema de la dérivée ne sont pas identifiés.')
    Pmin = P_EoS(T,x[0]*b-c,a,b,c)
    Pmax = P_EoS(T,x[1]*b-c,a,b,c)
    if not np.all(np.isfinite([Pmin,Pmax])) or Pmax <= max(Pmin,0.):
        raise RuntimeError('Bornes de pression non physiques.')
    return max(Pmin,0.), Pmax

# Procédure initiale : balayage volumique (secours).
#==============================================================================
def Bornes_Psat_balayage(T,a,c):
    bnew = b-c # covolume corrigé
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
        dP = Pnew-Pold
        Pold = Pnew
        it += 1
        if it > itmax:
            raise RuntimeError('Balayage de Pmin : trop d’itérations.')
    Pmin = max(Pold,0.)

    # Recherche de Pmax :
    it = 0
    while dP>0:
        vold = vnew
        vnew = vold + stepv
        Pnew = P_EoS(T,vnew,a,b,c)
        dP = Pnew-Pold
        Pold = Pnew
        it += 1
        if it > itmax:
            raise RuntimeError('Balayage de Pmax : trop d’itérations.')
    Pmax = Pold
    if not np.all(np.isfinite([Pmin,Pmax])) or Pmax <= Pmin:
        raise RuntimeError('Balayage : bornes de pression invalides.')
    return Pmin, Pmax

# Dichotomie sur la pression avec contrôle des volumes des deux phases.
#==============================================================================
def Dichotomie_Psat(T,a,c,Pmin,Pmax):
    if Pmax <= Pmin or Pmax <= 0.:
        raise RuntimeError('Intervalle de saturation invalide.')
    for i in range(100):
        P0 = 0.5*(Pmin+Pmax)
        if P0 <= 0.:
            raise RuntimeError('Pression de saturation non positive.')
        VolMol = resolEoS(T,P0,a,b,c)
        if len(VolMol) < 2:
            raise RuntimeError('Deux phases distinctes introuvables.')
        volLiq = min(VolMol)
        volGaz = max(VolMol)
        if volGaz <= volLiq*(1.+1e-8):
            raise RuntimeError('Volumes liquide/vapeur confondus.')
        delta = lnfug_EoS(T,volLiq,a,b,c) - lnfug_EoS(T,volGaz,a,b,c)
        if not np.isfinite(delta):
            raise RuntimeError('Fugacités non finies.')
        if delta > 0.:
            Pmin = P0
        else:
            Pmax = P0
        if Pmax-Pmin <= eps*max(P0,1.):
            return P0,volLiq,volGaz
    raise RuntimeError('Dichotomie de saturation non convergée.')

# Calcul d'une pression de vapeur saturante à T fixée : 
#==============================================================================
def Psat(T,c):
    # Psat en Pa,
    # Inputs : T en K et c (correction volumique) en m3/mol
    # Outputs : Psat, vliq, vgaz (vliq et vgaz = vol. mol du liq et du gaz 
    #                             en équilibre LV, m3/mol)
    if T>=Tc:
        return 0.,0.,0.

    a, da, d2a = Coeff_a_EoS(T,Tc,w,0)
    erreur_derivee = None
    for nom, methode in [('dérivée',Bornes_Psat_derivee),
                         ('balayage de secours',Bornes_Psat_balayage)]:
        try:
            Pmin,Pmax = methode(T,a,c)
            resultat = Dichotomie_Psat(T,a,c,Pmin,Pmax)
            methodes_Psat[nom] += 1
            return resultat
        except (ValueError, ArithmeticError, RuntimeError) as erreur:
            if nom == 'dérivée':
                erreur_derivee = erreur
            else:
                raise RuntimeError('Équilibre LV non convergé à T = ' + str(T) +
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
    # c = Vliq EoS originale (Tr = 0.8) - Vliq exp (Tr = 0.8)
    cvol = veos_liq08 - vliq08
    bnew = b - cvol
    if bnew <= 0.:
        raise ValueError('Translation volumique non physique : b-c <= 0.')
    return cvol, bnew

# État stable et propriétés à T et P imposés.
#==============================================================================
def Calcul_etat(T,P,c):
    a,da,d2a = Coeff_a_EoS(T,Tc,w,2)
    volumes = resolEoS(T,P,a,b,c)
    if len(volumes) == 0:
        raise RuntimeError('Aucun volume molaire physique à T et P imposés.')

    psat = None
    if T < Tc:
        psat, vliq_sat, vgaz_sat = Psat(T,c)
        # Sur la courbe de saturation, la fraction vaporisée reste inconnue.
        if abs(P-psat) <= 1e-5*psat:
            phases = [('Saturated liquid',vliq_sat),
                      ('Saturated vapor',vgaz_sat)]
        elif P > psat:
            phases = [('Liquid',min(volumes))]
        else:
            phases = [('Vapor',max(volumes))]
    else:
        if np.isclose(T,Tc,rtol=0.,atol=1e-10) and \
                np.isclose(P,Pc,rtol=1e-7,atol=0.):
            nom_phase = 'Critical state'
        elif T > Tc and P > Pc:
            nom_phase = 'Supercritical fluid'
        else:
            nom_phase = 'Single-phase fluid'
        phases = [(nom_phase,max(volumes))]

    resultats = []
    cp0 = CpGP(CoeffCpGP,T)
    for nom_phase, v in phases:
        cp_res = Cpres_EoS(T,v,a,b,c,da,d2a)
        resultats.append({
            'phase': nom_phase,
            'v': v,               # m3/mol
            'rho': 1./v,          # mol/m3
            'Cp0': cp0,           # J/mol/K
            'Cpres': cp_res,      # J/mol/K
            'Cp': cp0+cp_res      # J/mol/K
        })
    return psat, resultats

#==============================================================================
#==============================================================================
# MAIN :
#
# Calcul de la correction volumique c :
#------------------------------------------------------------------------------
c, bnew = Calcul_c()
vcrit = vcrit - c # volume molaire critique après translation
if vcrit <= 0.:
    raise ValueError('Volume molaire critique translaté non physique.')

# Bilan console et sauvegarde du bilan dans un fichier texte.
#------------------------------------------------------------------------------
rapport = []
def afficher(texte=''):
    print(texte)
    rapport.append(texte)

afficher('='*70)
afficher('tc-PR : input data and calculated properties')
afficher('='*70)
afficher('INPUT DATA')
afficher(f'  T                  = {T:.6f} K')
afficher(f'  P                  = {P:.6f} Pa  ({P*1e-5:.6f} bar)')
afficher(f'  Tc                 = {Tc:.6f} K')
afficher(f'  Pc                 = {Pc:.6f} Pa  ({Pc*1e-5:.6f} bar)')
afficher(f'  Acentric factor    = {w:.8f}')
afficher(f'  VL(Tr=0.8)        = {vliq08:.10g} m3/mol  ({vliq08*1e6:.6f} cm3/mol)')
afficher('  DIPPR 127 (Cp0/R):')
for nom, valeur in zip('ABCDEFG',CoeffCpGP):
    afficher(f'    {nom} = {valeur:.10g}')

afficher('')
afficher('CALCULATED PROPERTIES')
afficher(f'  Volume translation c = {c:.10g} m3/mol')
afficher(f'  Critical molar volume = {vcrit:.10g} m3/mol')

psat_etat, etats = Calcul_etat(T,P,c)
if psat_etat is not None:
    afficher(f'  Psat(T)              = {psat_etat:.6f} Pa  ({psat_etat*1e-5:.6f} bar)')
for etat in etats:
    afficher('')
    afficher(f"  Physical state        = {etat['phase']}")
    afficher(f"  Molar volume          = {etat['v']:.10g} m3/mol")
    afficher(f"  Molar density         = {etat['rho']:.8f} mol/m3  ({etat['rho']*1e-3:.8f} mol/L)")
    afficher(f"  Ideal-gas Cp          = {etat['Cp0']:.8f} J/mol/K")
    afficher(f"  Residual Cp           = {etat['Cpres']:.8f} J/mol/K")
    afficher(f"  Total Cp              = {etat['Cp']:.8f} J/mol/K")
if len(etats) == 2:
    afficher('  NOTE: at saturation, mixture density and Cp require vapor quality.')

# Calcul et tracé de Psat(T)
#------------------------------------------------------------------------------
Tmin = 0.75*Tc # K
Tmax = np.floor(Tc) 
DT = 0.5
T_List = np.arange(Tmin, Tmax, DT)
Psat_List = np.full(len(T_List),np.nan)
Vliq_List = np.full(len(T_List),np.nan)
Vgaz_List = np.full(len(T_List),np.nan)
DvapH_List = np.full(len(T_List),np.nan)

CpLiq_List = []
T_ListCp = []
anomalies = []

for i in range(len(T_List)):
    Ti = T_List[i]
    try:
        res = Psat(Ti,c)
    except RuntimeError as erreur:
        anomalies.append(str(erreur))
        continue
    vl = res[1]
    vg = res[2]
    
    Psat_List[i] = res[0]*1e-5  # bar
    Vliq_List[i] = res[1]*1e6 # mL/mol
    Vgaz_List[i] = res[2]*1e6 # mL/mol
    
    # Calcul du param attractif a à la température T :
    a, da, d2a = Coeff_a_EoS(Ti,Tc,w,2)

    DvapH_List[i] = hres_EoS(Ti,vg,a,b,c,da) \
        - hres_EoS(Ti,vl,a,b,c,da)
    if Ti < 0.98*Tc:
        T_ListCp = np.append(T_ListCp,Ti)
        CpL = Cpres_EoS(Ti,vl,a,b,c,da,d2a) + CpGP(CoeffCpGP,Ti)
        CpLiq_List = np.append(CpLiq_List, CpL)

# Point critique : 
T_List = np.append(T_List,Tc)
Psat_List = np.append(Psat_List,Pc*1e-5) # bar
Vliq_List = np.append(Vliq_List,vcrit*1e6) # mL/mol
Vgaz_List = np.append(Vgaz_List,vcrit*1e6) # mL/mol
DvapH_List = np.append(DvapH_List,0.)

#************ TRACE DU PLAN (P, T)
nfig += 1
plt.figure(nfig)
#************
plt.clf()
plt.plot(T_List, Psat_List,'k')
plt.plot(Tc, Pc*1e-5,'ko')
plt.xlabel('$T$ (K)',fontsize=10)
plt.ylabel('$P^{sat}$ (bar)',fontsize=10)
plt.title('Vapor-pressure curve',fontsize=20)
plt.show()

#************ TRACE DU PLAN (T, densité)
nfig += 1
plt.figure(nfig)
#************
vcrit_cm3 = vcrit*1e6 # cm3/mol
vliq08_cm3 = vliq08*1e6
plt.clf()
plt.plot(1e3/Vliq_List , T_List,'b',label='Liquid branch')
plt.plot(1e3/Vgaz_List , T_List,'r',label='Vapor branch')
plt.plot(1e3/vcrit_cm3, Tc,'ko',label = 'Critical point')
plt.plot(1e3/vliq08_cm3, 0.8*Tc,'g*',label = r'$\rho_{liq}(T_r=0.8)$')
plt.xlabel(r'$\rho$ (mol/L)',fontsize=10)
plt.ylabel('$T$ (K)',fontsize=10)
plt.title(r'Vapor-liquid coexistence ($T$,$\rho$)',fontsize=20)
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
plt.title('Enthalpy of vaporization',fontsize=20)
plt.show()


#************ TRACE DU PLAN (T, cP liq)
nfig += 1
plt.figure(nfig)
#************
vcrit_cm3 = vcrit*1e6 # cm3/mol
vliq08_cm3 = vliq08*1e6
plt.clf()
plt.plot(T_ListCp, CpLiq_List,'b',label='Saturated liquid')
plt.xlabel(r'$T$ (K)',fontsize=10)
plt.ylabel(r'$c_{P,liq}$ (J/mol/K)',fontsize=10)
plt.title('Saturated-liquid heat capacity',fontsize=20)
plt.legend(fontsize=10)
plt.show()

afficher('')
afficher('SATURATION CURVE')
afficher(f"  Computed points       = {int(np.sum(np.isfinite(Psat_List)))-1}")
afficher(f"  Derivative method     = {methodes_Psat['dérivée']} calls")
afficher(f"  Volume-scan fallback  = {methodes_Psat['balayage de secours']} calls")
afficher(f"  Failed temperatures   = {len(anomalies)}")
if anomalies:
    afficher('  First failure: ' + anomalies[0])

sortie = Path(__file__).resolve().parent / (
    'tcPR_resultats_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.txt')
afficher('')
afficher('Result file: ' + str(sortie))
sortie.write_text('\n'.join(rapport)+'\n',encoding='utf-8')
