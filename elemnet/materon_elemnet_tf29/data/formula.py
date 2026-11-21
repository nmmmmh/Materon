from __future__ import absolute_import, division, print_function

import re
from typing import Dict, List

import numpy as np

ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
    "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
SYMBOL_TO_Z = {symbol: index + 1 for index, symbol in enumerate(ELEMENT_SYMBOLS)}
_SIMPLE_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*(?:[eE][+-]?[0-9]+)?)")


def _simple_parse(formula: str) -> Dict[str, float]:
    """Fallback parser for flat formulae; formal cache uses pymatgen."""
    formula = str(formula).strip()
    if not formula:
        raise ValueError("Empty composition")
    position = 0
    amounts = {}
    for match in _SIMPLE_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError("Fallback parser cannot parse formula %r" % formula)
        symbol = match.group(1)
        if symbol not in SYMBOL_TO_Z:
            raise ValueError("Unknown element symbol %r" % symbol)
        amount_text = match.group(2)
        amount = float(amount_text) if amount_text else 1.0
        amounts[symbol] = amounts.get(symbol, 0.0) + amount
        position = match.end()
    if position != len(formula) or not amounts:
        raise ValueError("Fallback parser cannot parse formula %r" % formula)
    return amounts


def composition_amounts(formula: str) -> Dict[str, float]:
    try:
        from pymatgen.core import Composition
        values = Composition(str(formula)).get_el_amt_dict()
        return {str(symbol): float(amount) for symbol, amount in values.items()}
    except ImportError:
        return _simple_parse(formula)


def formula_to_fraction_vector(formula: str, max_atomic_number: int = 118) -> np.ndarray:
    if max_atomic_number < 1 or max_atomic_number > 118:
        raise ValueError("max_atomic_number must be between 1 and 118")
    amounts = composition_amounts(formula)
    vector = np.zeros(max_atomic_number, dtype=np.float32)
    total = 0.0
    for symbol, amount in amounts.items():
        atomic_number = SYMBOL_TO_Z.get(symbol)
        if atomic_number is None:
            raise ValueError("Unknown element %r in %r" % (symbol, formula))
        if atomic_number > max_atomic_number:
            raise ValueError("Element %s exceeds max_atomic_number=%d" % (symbol, max_atomic_number))
        if amount < 0:
            raise ValueError("Negative amount in %r" % formula)
        vector[atomic_number - 1] += float(amount)
        total += float(amount)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Composition has non-positive total: %r" % formula)
    vector /= np.float32(total)
    return vector


def build_fraction_matrix(formulas: List[str], max_atomic_number: int = 118) -> np.ndarray:
    return np.stack([formula_to_fraction_vector(value, max_atomic_number) for value in formulas]).astype(np.float32)
