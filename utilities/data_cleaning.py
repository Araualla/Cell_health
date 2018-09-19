from utilities.constants import TREAT, CONC
from utilities.counts import count_cells_per_well, normalise_count_cells

# labels for concentration of treatments in the experiment
number2conc = {2: '0 ug/mL',
               3: '0.137 ug/mL',
               4: '0.412 ug/mL',
               5: '1.235 ug/mL',
               6: '3.704 ug/mL',
               7: '11.11 ug/mL',
               8: '33.33 ug/mL',
               9: '100 ug/mL',
               10: '300ug/mL'}
# labels for the nanoparticle treatments in the experiment
row2np = {'A': 'Si-F8BT',
          'B': 'Si-CNPPV',
          'C': 'Si-P3',
          'D': 'Si-P4',
          'E': 'PP-F8BT',
          'F': 'PP-CNPPV',
          'G': 'PP-P3',
          'H': 'PP-P4'}
# labels for the control treatments in the experiment
controls = {'A': 'FCCP Control',
            'B': 'FCCP Control',
            'C': 'Triton-X',
            'D': 'Triton-X',
            'E': 'H2O',
            'F': 'H2O',
            'G': 'DMSO',
            'H': 'DMSO'}


def clean_data(data):
    """Clean a csv file"""
    # removing Weighted_Relative_Moment_Inertia
	# high frequency of nan
    data.drop(columns=['Weighted_Relative_Moment_Inertia'])
    data.columns = [format_column_name(x) for x in data.columns]
    data = label_data(data)
    data = normalise_data(data)
    count = count_cells_per_well(data)
    normalised_counts = normalise_count_cells(data, count)
    return data, count, normalised_counts


def label_data(data):
    """ Takes one dataframe and applies the correct labels to each row"""
    # some rows miss these two features, which are fundamental. **EXTERMINATE**

    drop = data[['Area Nuc', 'Area Cell']].isnull().sum(axis=1) != 0
    drop = data.index.values[drop]
    data = data.drop(index=drop)

    data[CONC] = data.apply(lambda x: number2conc.get(x['Number'], 'control'), axis=1)
    data.head()

    data[TREAT] = data.apply(lambda x: row2np.get(x['Row'], 'control'), axis=1)
    data.head()

    for key in controls:
        data.loc[(data[CONC] == 'control') & (data['Row'] == key), TREAT] = controls[key]
    data = data.drop(columns=['Number', 'Count Nuc'])
    return data


def format_column_name(string):
    """Automatically reformats feature names into something more machine-readable."""
    string = ' '.join(string.strip().split())
    string = (string
              .replace('_', ' ')
              .replace('[', '')
              .title()
              .replace('- Um', '')
              )
    #     if ('Feret' in string or 'Perimeter' in string) and '(μm)' not in string:
    #         string += ' (μm)'
    if 'Mempernuc' in string:
        string = string.replace('Mempernuc', 'Mem Per Nuc')
    if 'Mitoint' in string:
        string = string.replace('Mitoint', 'Mito Int ')
        string = string.title()
    if 'dxa' in string or 'Dxa' in string:
        string = string.replace('dxa', ' DxA')
        string = string.replace('Dxa', ' DxA')
    if 'Wmoi' in string:
        string = string.replace('Wmoi', 'WMOI')
    if 'Conc' in string:
        string = string.replace('Conc', 'Concentration')
    return string


def format_dataframe_columns(df):
    df.columns = [format_column_name(colname) for colname in df.columns]
    return df


def normalise_data(data):
    """Z-scores all numeric data."""
    # select only numeric data
    numeric = data._get_numeric_data()
    # apply transformation
    numeric = numeric - numeric.mean()
    numeric = numeric / numeric.std()
    # mind that we don't have the classes column in this dataframe!
    # put class information back in
    numeric[CONC] = data[CONC].tolist()
    numeric[TREAT] = data[TREAT].tolist()
    return numeric
