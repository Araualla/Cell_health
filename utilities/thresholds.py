import numpy as np
import pandas as pd
from utilities.constants import TREAT, CONC, index_order, column_order


_THRESHOLD_ABOVE = 1
_THRESHOLD_BELOW = 0


def threshold_dict(data, _THRESHOLD):
    # TODO -- document
    """COMMENT"""
    # std and mean by column
    numerical_cols = data._get_numeric_data().columns
    negative_controls = data[data[CONC] == '0 ug/mL']
    std = np.array([negative_controls[col_name].std() for col_name in numerical_cols])
    mean = np.array([negative_controls[col_name].mean() for col_name in numerical_cols])
    target_count = np.array([negative_controls[col_name].count() for col_name in numerical_cols])
    # make an array which has one std or mean for each column of negative controls
    # Define threshold
    if _THRESHOLD == _THRESHOLD_ABOVE:
        threshold = mean + 2 * std
    elif _THRESHOLD == _THRESHOLD_BELOW:
        threshold = mean - 1 * std

    threshold_dict = {name: t for name, t in zip(numerical_cols, threshold)}
    # thresholds are calculated on negative controls
    return threshold_dict


def extract_threshold_percentile(data, feature, threshold_dict, _THRESHOLD):
    # TODO -- document
    """COMMENT"""
    result = dict()
    for t in set(data[TREAT]):
        conc_result = dict()
        for c in set(data[CONC]):
            selection = (data[CONC] == c) & (data[TREAT] == t)
            if _THRESHOLD == _THRESHOLD_ABOVE:
                conc_result[c] = (data.loc[selection, :][feature] > threshold_dict[feature]).sum() / selection.sum()
            elif _THRESHOLD == _THRESHOLD_BELOW:
                conc_result[c] = (data.loc[selection, :][feature] < threshold_dict[feature]).sum() / selection.sum()
        result[t] = conc_result
    return pd.DataFrame.from_dict(result, orient='index').loc[index_order, column_order]


def extract_from_replicate(data, _THRESHOLD):
    # TODO -- document
    """COMMENT"""
    thresholds = threshold_dict(data, _THRESHOLD)
    feature_list = list(data.keys())
    feature_list = [f for f in feature_list if f not in [TREAT, CONC]]
    threshold_percentile = dict()
    for feature in feature_list:
        threshold_percentile[feature] = extract_threshold_percentile(data, feature, thresholds, _THRESHOLD)
    return threshold_percentile


# def extract_threshold_percentile_by_feature (data, extract_threshold_percentile, threshold_dict):
#     percentile_by_feature = dict()
#     feature_list = list(data.keys())
#     feature_list = [f for f in feature_list if f not in [TREAT, CONC]]
#     for f in feature_list:
#         percentile_by_feature[f] =extract_threshold_percentile
#     return percentile_by_feature
