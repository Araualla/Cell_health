import numpy as np
from utilities.constants import TREAT, CONC


def count_cells_per_well(data):
    """Counts number of cells per treatment and per concentration (i.e. per well)"""
    results = dict()
    for t in set(data[TREAT]):
        count = dict()
        for c in set(data[CONC]):
            selection = data[TREAT] == t
            selection = data.loc[selection, CONC] == c
            count[c] = selection.sum()
        results[t] = count
    return results


def normalise_count_cells(data, count):
    """Normalise cell counts with respect to the number of cells in the controls"""
    # select nuber of cells in negative control wells, excluding if 0
    n_in_controls = [count[t]['0 ug/mL'] for t in count if count[t]['0 ug/mL']]
    c, deltac = np.mean(n_in_controls), np.std(n_in_controls)
    # treatment_count number of cells per treatment and per concentration (i.e. per well)
    normalised = dict()
    for t in set(data[TREAT]):
        treatment_count = dict()
        for k in set(data[CONC]):
            selection = data[TREAT] == t
            selection = data.loc[selection, CONC] == k
            treatment_count[k] = selection.sum() / c * 100
        normalised[t] = treatment_count
    return normalised
