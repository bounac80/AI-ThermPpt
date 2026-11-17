# Example: tc-PR calculation for 1-propanol

`tcPR_Calcul_Cp.py` is a stand-alone script (only `numpy` and `matplotlib` are needed) that computes the properties of a pure compound with the volume-translated Peng-Robinson equation of state (tc-PR). The molecule used as example is **1-propanol** (SMILES `CCCO`).

## 1. Input data

All inputs are set at the beginning of the script:

| Variable | Meaning | Unit | 1-propanol |
|---|---|---|---|
| `T` | temperature of the studied state | K | 298.15 |
| `P` | pressure of the studied state | Pa | 1.0e5 |
| `Tc` | critical temperature | K | 536.63 |
| `Pc` | critical pressure | Pa | 51.71e5 |
| `w` | acentric factor | - | 0.6211 |
| `vliq08` | saturated-liquid molar volume at Tr = 0.8 | m³/mol | 92.99e-6 |
| `CoeffCpGP` | the seven coefficients A-G of the ideal-gas Cp°/R correlation | A, B, D, F: -; C, E, G: K | `[3.3236, 3.0552, 349.63, 13.677, 1193.3, 11.669, 3460.0]` |

For another molecule, replace these values by the ones predicted by the AI tools:

- `Tc`, `Pc`, `w` and `vliq08`: <https://thermokinetics-lrgp.com/thermppt/>. The site gives *Pc* in bar and the liquid volume in cm³/mol: convert them to Pa and m³/mol.
- `CoeffCpGP`: <https://thermokinetics-lrgp.com/cp-versus-temp/>. The coefficients displayed there describe *Cp°/R*.

The same predictions can also be obtained locally with the models provided in `../models/` (see the main README of `tcPR_Calcul`).

## 2. Run

```bash
pip install numpy matplotlib
python tcPR_Calcul_Cp.py
```

## 3. Results

- Console output (also saved in a time-stamped text file next to the script, `tcPR_resultats_<date>.txt`):
  - the input data;
  - the volume translation *c* and the translated critical volume;
  - the vapor pressure at `T`;
  - the stable phase at `(T, P)`, its molar volume and density, the ideal-gas, residual and total heat capacities.
- Four figures, drawn between 0.75·Tc and Tc:
  1. vapor-pressure curve;
  2. vapor-liquid coexistence in the (T, ρ) plane, with the critical point and the liquid density at Tr = 0.8;
  3. enthalpy of vaporization;
  4. heat capacity of the saturated liquid.

Reference output for the 1-propanol example (T = 298.15 K, P = 1 bar):

```
Volume translation c  = 8.72964744e-07 m3/mol
Critical molar volume = 0.000264367822 m3/mol
Psat(T)               = 3472.914553 Pa
Physical state        = Liquid
Molar density         = 13.19651572 mol/L
Ideal-gas Cp          = 84.97048740 J/mol/K
Residual Cp           = 73.31838353 J/mol/K
Total Cp              = 158.28887093 J/mol/K
```

## 4. Method

- Peng-Robinson cubic equation of state with the α-function `a(T) = a_c·[1 + m(ω)(1 − √(T/Tc))]²` and `m(ω) = 0.37464 + 1.54226·ω − 0.26992·ω²`.
- Constant volume translation `c = v_liq,EoS(Tr = 0.8) − vliq08`.
- Vapor pressure from the equality of the fugacities of the two phases. The bounds on the pressure come from the extrema of the isotherm; a volume scan is used as a fall-back.
- Heat capacity: `Cp = Cp° + Cp,res`, with `Cp°` from the seven-coefficient correlation (`R` times the *Cp°/R* correlation) and `Cp,res` the residual part given by the equation of state.

## 5. Reference

If you use this calculation, please cite the publications listed in the main README of `tcPR_Calcul`.

Contact: roda.bounaceur@univ-lorraine.fr
