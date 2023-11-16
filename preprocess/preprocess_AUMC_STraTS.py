import pandas as pd
import numpy as np
import os

import shutil
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pickle
import re

adb_path = ''
admissions = pd.read_csv(os.path.join(adb_path, 'admission.csv'))
oc = {'RecordID': [], 'length_of_stay': [], 'in_hospital_mortality': []}

def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d

ts = []
for patient in tqdm(os.listdir(os.path.join(adb_path, 'processed'))):
    if patient == 'Amsterdam-Mimic.xlsx':
        continue
    entries = admissions[admissions['patientid']==int(patient)]
    for i in range(len(entries)):
        entry = entries[entries['admissioncount']==i+1]
        if entry.iloc[0]['lengthofstay']<48:
            continue
        # morts.append(int(entry['destination']=='Overleden'))

        data = pd.read_csv(os.path.join(adb_path, 'processed', patient, 'episode{}_full_events_timeseries.csv'.format(i+1)))
        keys = list(data.keys()).remove('Hours')
        if i > 1:
            data['Hours'] -= data['Hours'].min()
        else:
            data = data[data['Hours']>0]
        data = data[data['Hours']<48]
        if len(data) <5:
            continue
        data = pd.melt(data, id_vars=['Hours'], value_vars=keys, var_name= 'Parameter', value_name='Value').dropna(how = 'any')

        age = str(entry.iloc[0]['agegroup'])
        # if age.__contains__('+'):
        #     age = float(age[:-1])
        # else:
        #     age = float(age.split('-')[0])
        age = float(re.search(r'[0-9]+', age).group())

        # height = str(entry.iloc[0]['heightgroup'])
        # if height.__contains__('+'):
        #     height = float(height[:-1])
        # else:
        #     height = float(height.split('-')[0])
        gender = float(entry.iloc[0]['gender'] == 'Man')
        if np.isnan(age) or np.isnan(gender):
            print('Error')
            print(str(entry.iloc[0]['agegroup']), entry.iloc[0]['gender'])
            exit()
        data.loc[len(data)] = {'Hours': 0, 'Parameter': 'Age', 'Value': age}
        data.loc[len(data)] = {'Hours': 0, 'Parameter': 'Gender', 'Value': gender}
        # data.loc[len(data)] = {'Hours': 0, 'Parameter': 'Height', 'Value': height}

        data['RecordID'] = '{}#{}'.format(patient, i)
        ts.append(data.sort_values(by = 'Hours'))

        mort = int(entry['destination']=='Overleden')
        oc['RecordID'].append('{}#{}'.format(patient, i))
        oc['in_hospital_mortality'].append(mort)
        oc['length_of_stay'].append(entry.iloc[0]['lengthofstay'])
    
print(len(ts), [len(oc[i]) for i in oc.keys()])

ts = pd.concat(ts)
ts = ts.dropna(how = 'any')
oc = pd.DataFrame(oc)
ts.rename(columns={'Hours':'hour', 'Parameter':'variable', 'Value':'value'}, inplace=True)
rec_ids = sorted(list(ts.RecordID.unique()))
rid_to_ind = inv_list(rec_ids)
oc = oc.loc[oc.RecordID.isin(rec_ids)]
ts['ts_ind'] = ts.RecordID.map(rid_to_ind)
oc['ts_ind'] = oc.RecordID.map(rid_to_ind)

train_oc, test_oc = train_test_split(oc, test_size=0.2, stratify=oc['in_hospital_mortality'], random_state=13)
train_oc, val_oc = train_test_split(train_oc, test_size=0.2, stratify=train_oc['in_hospital_mortality'], random_state=13)

train_oc['RecordID'].to_csv('./data/AUMC_train_idx.csv')
val_oc['RecordID'].to_csv('./data/AUMC_val_idx.csv')
test_oc['RecordID'].to_csv('./data/AUMC_test_idx.csv')

train_ind = oc.loc[oc.RecordID.isin(train_oc['RecordID'])].ts_ind
valid_ind = oc.loc[oc.RecordID.isin(val_oc['RecordID'])].ts_ind
test_ind = oc.loc[oc.RecordID.isin(test_oc['RecordID'])].ts_ind

# ts.drop(columns='RecordID', inplace=True)
# oc.drop(columns='RecordID', inplace=True)
ts = ts.drop_duplicates()
ts = ts.dropna(how = 'any')
print(np.sum(np.isnan(ts['value'].to_numpy().astype('float32'))))

# print(ts[ts['variable'] == 'Asparate aminotransferase'])

# means_stds = ts.groupby('variable').agg({'value':['mean', 'std']})

means = ts.groupby('variable').quantile([.50])
stds = ts.groupby('variable').quantile([.95, .05])
means = means.reset_index().loc[:, ['variable', 'value']].rename(columns={'value': 'mean'})
a = stds.reset_index().loc[stds.reset_index()['level_1'] == 0.95, ['variable', 'value']].set_index('variable')
b = stds.reset_index().loc[stds.reset_index()['level_1'] == 0.05, ['variable', 'value']].set_index('variable')
stds = a.sub(b).reset_index().rename(columns={'value': 'std'})

means_stds = means.merge(stds, on = 'variable', how = 'left')
means_stds.loc[means_stds['std']==0, 'std'] = 1
print(means_stds)
ts = ts.merge(means_stds.reset_index(), on='variable', how='left')
ii = ts.variable.apply(lambda x:not(x.startswith('ICUType')))&(~ts.variable.isin(['Age', 'Gender']))
ts.loc[ii, 'value'] = (ts.loc[ii, 'value']-ts.loc[ii, 'mean'])/ts.loc[ii, 'std']
# ts.loc[ii, 'value'] = ts.loc[ii, 'value'].clip(lower=-2, upper = 2)
maxv = max(ts['value'])
print(maxv)
print(len(ts))
index = ts[(~ts.value.between(-2, 2))&(~ts.variable.isin(['Age', 'Gender']))].index
print(len(index))
ts.drop(index, inplace=True)
print(len(ts))
maxv = max(ts['value'])
print(maxv)
print(np.percentile(ts['value'], 99))
pickle.dump([ts, oc, train_ind, valid_ind, test_ind], open('AUMC_preprocessed_clipped.pkl','wb'))
