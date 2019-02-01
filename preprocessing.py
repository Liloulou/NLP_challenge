import pandas as pd
import numpy as np
import time

final_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
letter = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
]

cim_vocab_list = [''.join([i, j, k, l]) for i in letter for j in number for k in number for l in final_number]

raw_characters = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',',
       '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       ':', '<', '=', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
       'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
       'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b',
       'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
       'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|',
       '}', '\x80', '\x85', '\x8c', '\x92', '\x95', '\x96', '\x9c',
       '\xa0', '§', '¨', '«', '°', '±', '²', 'µ', '»', 'Â', 'Ä', 'È', 'É',
       'Ê', 'Ë', 'Î', 'Ï', 'Ö', 'Û', 'Ü', 'à', 'â', 'ä', 'ç', 'è', 'é',
       'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', 'ü']

delimiter_characters = [
       ',', '/', '|', '§'
]

to_space_characters = [
       '"', '#', '$', '%', '&', "'", '(', ')', '{', '}', '<', '´',
       '=', '>', '*', '+', '`', '[', ']', '_', '-', '.', '’', '•', '~'
]

del_characters = [
       '\\', '\x80', '\x85', '\x8c', '\x92', '\x95', '\x96', '\x9c', '\xa0', '¨', '«', '»', '°', '±', '²', 'µ', '^', ':'
]

replace_dict = {
       'a': ['à', 'â', 'ä'],
       'c': ['ç'],
       'e': ['è', 'é', 'ê', 'ë'],
       'i': ['î', 'ï', 'ì'],
       'o': ['ô', 'ö', 'ó'],
       'u': ['ù', 'û', 'ü'],
       'A': ['Â', 'Ä', 'À'],
       'E': ['È', 'É', 'Ê', 'Ë'],
       'I': ['Î', 'Ï'],
       'O': ['Ö', 'Ô', 'Ó'],
       'U': ['Û', 'Ü'],
       'oe': ['œ'],
       ' ': to_space_characters,
       '': del_characters,
       '/': delimiter_characters
}


def collapse_columns(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.to_series().apply(lambda x: "_".join(x))
    return df


def merge_and_clean(brut, calc):

    brut = brut.loc[~brut['TexteLigneCause'].isna()]
    brut = brut.loc[brut['TexteLigneCause'] != '!']
    brut = brut.loc[~(~brut[['Unnamed: ' + str(x) for x in range(6, 13)]].isna()).any(1)]
    brut = brut.drop(
        columns=[
            'TypeIntervalleLigneCause', 'ValeurIntervalle', 'LAST_CHANGED_DATETIME',
            'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10',
            'Unnamed: 11', 'Unnamed: 12'
        ]
    )

    calc = calc.loc[~calc['CodeCIM'].isna()]
    calc = calc.drop(columns=['TexteCause', 'TypeCode', 'LAST_CHANGED_DATETIME'])

    text_arr = np.array(brut['TexteLigneCause'])

    clean_arr = np.array(text_arr)

    for key in replace_dict.keys():
        for elem in replace_dict[key]:
            clean_arr = [x.replace(elem, key) for x in clean_arr]

    clean_arr = [" ".join(x.split()) for x in clean_arr]

    is_upper = [''.join([str(x.isupper()*1) for x in y]) for y in clean_arr]

    clean_arr = [x.lower() for x in clean_arr]

    brut['RawText'] = clean_arr
    brut['IsUpper'] = is_upper
    brut = brut.drop(columns=['TexteLigneCause'])

    flat_calc = calc.set_index(['NumCertificat', 'LigneCause', 'RangLigneCause']).unstack('RangLigneCause')
    flat_calc = collapse_columns(flat_calc)

    flat_brut = brut.set_index(['NumCertificat', 'LigneCause'])

    df = flat_brut.merge(flat_calc, how='inner', left_index=True, right_index=True)

    df[['CodeCIM_' + str(i) for i in range(1, 18)]] = df[
        ['CodeCIM_' + str(i) for i in range(1, 18)]
    ].replace(np.nan, '')

    df['ICD10'] = df[
        ['CodeCIM_' + str(i) for i in range(1, 18)]
    ].apply(lambda x: ' '.join(x).replace('  ', ''), axis=1)
    df['ICD10'].loc[df['ICD10'].str.slice(-1) == ' '] = df['ICD10'].loc[
        df['ICD10'].str.slice(-1) == ' '
    ].str.slice(0, -1)

    df = df.drop(columns=['CodeCIM_' + str(i) for i in range(1, 18)])
    df = df[['RawText', 'ICD10', 'IsUpper']]

    df = pd.DataFrame(np.array(df), columns=['RawText', 'ICD10', 'IsUpper'])

    df = df.loc[df['RawText'].str.len() > 5]

    return df


def clean(df):

    clean_df = df.drop(
        columns=['YearCoded', 'Gender', 'Age', 'LocationOfDeath', 'IntType', 'IntValue', 'StandardText']
    )
    clean_df = clean_df.loc[~clean_df.isna().any(1)]
    clean_df['CauseRank'] = clean_df['CauseRank'].str.slice(2)
    duplicate = clean_df['DocID'].loc[clean_df[['DocID', 'LineID', 'CauseRank']].duplicated()].unique()
    clean_df = clean_df.loc[~clean_df['DocID'].isin(duplicate)]
    clean_df = clean_df.set_index(['DocID', 'LineID', 'CauseRank']).unstack('CauseRank')
    clean_df = collapse_columns(clean_df)

    num_codes = clean_df.shape[1] // 2
    clean_df = clean_df.replace(np.nan, '')
    clean_df['RawText'] = clean_df['RawText_1']

    for i in range(1, num_codes):
        clean_df['RawText'].loc[clean_df['RawText'] == ''] = clean_df[
            'RawText_' + str(i + 1)
        ].loc[clean_df['RawText'] == '']

    clean_df = clean_df.drop(columns=['RawText_' + str(i+1) for i in range(num_codes)])

    clean_df['ICD10'] = clean_df[
        ['ICD10_' + str(i+1) for i in range(num_codes)]
    ].apply(lambda x: ' '.join(x).replace('  ', ''), axis=1)
    clean_df['ICD10'].loc[clean_df['ICD10'].str.slice(-1) == ' '] = clean_df['ICD10'].loc[
        clean_df['ICD10'].str.slice(-1) == ' '
        ].str.slice(0, -1)

    clean_df = clean_df.drop(columns=['ICD10_' + str(i+1) for i in range(num_codes)])

    clean_arr = np.array(clean_df['RawText'])

    for key in replace_dict.keys():
        for elem in replace_dict[key]:
            clean_arr = [x.replace(elem, key) for x in clean_arr]

    clean_arr = [" ".join(x.split()) for x in clean_arr]
    is_upper = [''.join([str(x.isupper() * 1) for x in y]) for y in clean_arr]
    clean_arr = [x.lower() for x in clean_arr]

    clean_df['RawText'] = clean_arr
    clean_df['IsUpper'] = is_upper

    clean_df = clean_df.loc[clean_df['RawText'].str.len() > 5]
    clean_df['ICD10'].loc[clean_df['ICD10'].str.len() == 3] += '.'

    return clean_df


def clean_dictionary(df):

    clean_df = df[df.columns[:2]]
    clean_df.columns = ['RawText', 'ICD10']

    clean_arr = np.array(clean_df['RawText'])

    for key in replace_dict.keys():
        for elem in replace_dict[key]:
            clean_arr = [x.replace(elem, key) for x in clean_arr]

    clean_arr = [" ".join(x.split()) for x in clean_arr]
    is_upper = [''.join([str(x.isupper() * 1) for x in y]) for y in clean_arr]
    clean_arr = [x.lower() for x in clean_arr]

    clean_df['RawText'] = clean_arr
    clean_df['IsUpper'] = is_upper

    clean_df['ICD10'].loc[clean_df['ICD10'].str.len() > 4] = clean_df['ICD10'].loc[
        clean_df['ICD10'].str.len() > 4
    ].str.slice(0, 4)
    clean_df = clean_df.loc[clean_df['RawText'].str.len() > 5]
    clean_df = clean_df.loc[~clean_df.isna().any(1)]

    return clean_df


def make_icdlib_from_text_file(text_file):

    data = np.loadtxt(text_file, dtype=np.str, delimiter='|')
    icd10 = np.array([x[:4].replace(' ', '') for x in data[:, 0]])

    clean_arr = data[:, 3]

    for key in replace_dict.keys():
        for elem in replace_dict[key]:
            clean_arr = [x.replace(elem, key) for x in clean_arr]

    clean_arr = [" ".join(x.split()) for x in clean_arr]
    is_upper = [''.join([str(x.isupper() * 1) for x in y]) for y in clean_arr]
    clean_arr = [x.lower() for x in clean_arr]

    df = pd.DataFrame({'RawText': clean_arr, 'ICD10': icd10, 'IsUpper': is_upper})

    clean_arr = data[:, 2]

    for key in replace_dict.keys():
        for elem in replace_dict[key]:
            clean_arr = [x.replace(elem, key) for x in clean_arr]

    clean_arr = [" ".join(x.split()) for x in clean_arr]
    is_upper = [''.join([str(x.isupper() * 1) for x in y]) for y in clean_arr]
    clean_arr = [x.lower() for x in clean_arr]

    df_2 = pd.DataFrame({'RawText': clean_arr, 'ICD10': icd10, 'IsUpper': is_upper})

    df = pd.concat([df, df_2])
    df = df.loc[df['RawText'].str.len() > 5]

    return df


def get_unique(df):
    return np.unique(list(''.join(df['RawText'])))


print('beginning first dataset preprocessing')
start = time.time()
df_1 = clean(pd.read_csv('data/raw/AlignedCauses_2006-2012.csv', encoding='utf-8', dtype=np.str, delimiter=';'))
print(time.time() - start)
print(get_unique(df_1))
print('beginning second dataset preprocessing')
start = time.time()
df_2 = clean(pd.read_csv('data/raw/AlignedCauses_2013full.csv', encoding='utf-8', dtype=np.str, delimiter=';'))
print(time.time() - start)
print(get_unique(df_2))
print('beginning third dataset preprocessing')
start = time.time()
df_3 = clean(pd.read_csv('data/raw/AlignedCauses_2014_full.csv', encoding='utf-8', dtype=np.str, delimiter=';'))
print(time.time() - start)
print(get_unique(df_3))
print('beginning fourth dataset preprocessing')
start = time.time()
brut_df = pd.read_csv('data/raw/CausesBrutes2015.csv', encoding='latin-1', dtype=np.str, delimiter=';')
calc_df = pd.read_csv('data/raw/CausesCalculees2015.csv', encoding='latin-1', dtype=np.str, delimiter=';')

df_4 = merge_and_clean(brut_df, calc_df)
print(time.time() - start)
print(get_unique(df_4))

dataset = pd.concat([df_1, df_2, df_3, df_4])

print('shuffling dataset')
dataset = dataset.sample(frac=1)  # shuffle dataset

train = dataset.iloc[:-100000]
valid = dataset.iloc[-100000:-50000]
test = dataset.iloc[-50000:]

dictionary = clean_dictionary(pd.read_csv('data/raw/Dictionnaire2015.csv', encoding='utf-8', delimiter=';'))
lib = make_icdlib_from_text_file('data/raw/LIBCIM10.TXT')
train = pd.concat([train, dictionary, lib]).sample(frac=1)

train.to_csv('NLP_full_challenge_train.csv', index=False)
valid.to_csv('NLP_full_challenge_valid.csv', index=False)
test.to_csv('NLP_full_challenge_test.csv', index=False)
"""
calc.loc[calc['NumCertificat']=='ER20152150571']


# one_string = ''.join(text_arr)
one_string = ''.join(text_arr)
clean_string = ''.join(clean_arr)
clean = np.array(list(clean_string))
one = np.array(list(one_string))
print(np.unique(clean))
mean = []
for key in replace_dict.keys():
    print(key)
    for elem in replace_dict[key]:
        mean.append(np.mean(clean[np.argwhere(one == elem)] == key))

print(np.mean(mean))
"""
