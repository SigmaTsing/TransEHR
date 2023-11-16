import pandas as pd
import numpy as np
import os

import shutil
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pickle
import re

pheno_path = ''

def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d

ts = []
oc = {'RecordID': [], 'Phenotype': []}


entry_path = os.path.join('','mimic2physionet.csv')
entry = pd.read_csv(entry_path)
# table_events,column_events,feature_physionet,confidence
fts = entry.loc[entry['confidence'].isin([1, 2])]
column_1 = fts.loc[fts['table_events'] == 'full_events_timeseries']
column_2 = fts.loc[fts['table_events'] == 'full_events2_timeseries']
# mimic_param = zip(ft_con2['table_events'].tolist(), ft_con2['column_events'].tolist())
fts1 = column_1['column_events'].tolist()
fts2 = column_2['column_events'].tolist()
fts2.remove('Oxygen saturation')
vital_fts = ['Diastolic blood pressure', 'Glucose', 'Heart Rate', 'Mean blood pressure', 'Respiratory rate', 'Systolic blood pressure', 'Temperature']

adm = pd.read_csv('')
# print(listfile.keys())
train_ids = []
test_ids = []
for suffix in ['train', 'test']:
    listfile = pd.read_csv(os.path.join(pheno_path, suffix, 'listfile.csv'))
    for patient in tqdm(listfile['stay']):
        entry = listfile[listfile['stay'] == patient]
        
        id = patient.split('_')[0]
        epis = patient.split('_')[1].split('e')[-1]
        
        e1_path = os.path.join(pheno_path, suffix, '{}_episode{}_full_events_timeseries.csv'.format(int(id), int(epis)))
        e2_path = os.path.join(pheno_path, suffix, '{}_episode{}_full_events2_timeseries.csv'.format(int(id), int(epis)))
        if not os.path.exists(e1_path) or not os.path.exists(e2_path):
            print(patient)
            continue

        data1 = pd.read_csv(e1_path)
        data2 = pd.read_csv(e2_path)
        vital = pd.read_csv(os.path.join(pheno_path, suffix, patient))
        vital = vital.loc[:, ['Hours'] + vital_fts]
        data1 = data1.loc[:, ['Hours'] + fts1]
        data2 = data2.loc[:, ['Hours'] + fts2]

        merged = pd.merge(left = data1, right = data2, how = 'outer', )
        
        merged = pd.merge(data1, data2, how = 'outer', on = 'Hours').sort_values('Hours')
        merged = pd.merge(vital, merged, how = 'outer', on = 'Hours').sort_values('Hours')

        # print(merged.shape)
        # print(len(merged.keys()))
        # print(merged.keys())
        # exit()

        keys = list(merged.keys())
        keys.remove('Hours')
        # data = data[data['Hours']>0]
        # data = data[data['Hours']<48]
        data = pd.melt(merged, id_vars=['Hours'], value_vars=keys, var_name= 'Parameter', value_name='Value').dropna(how = 'any')

        static = adm[adm['SUBJECT_ID'] == int(id)]
        # print(static)
        age = static['AGE'].iloc[0]
        if static['GENDER'].iloc[0] == 'F':
            gender = 1
        else:
            gender = 0
        
        if np.isnan(age) or np.isnan(gender):
            print('Error')
            print(static)
            exit()
        data.loc[len(data)] = {'Hours': 0, 'Parameter': 'Age', 'Value': age}
        data.loc[len(data)] = {'Hours': 0, 'Parameter': 'Gender', 'Value': gender}

        data['RecordID'] = patient
        ts.append(data.sort_values(by = 'Hours'))

        pheno = entry.iloc[:, 2:].to_numpy()
        # print(pheno)
        oc['RecordID'].append(patient)
        oc['Phenotype'].append(pheno)

        if suffix == 'train':
            train_ids.append(patient)
        else:
            test_ids.append(patient)
    
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

train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=13)
train_oc = oc[oc['RecordID'].isin(train_ids)]
val_oc = oc[oc['RecordID'].isin(val_ids)]
test_oc = oc[oc['RecordID'].isin(test_ids)]

train_oc['RecordID'].to_csv('./data/Pheno_train_idx.csv')
val_oc['RecordID'].to_csv('./data/Pheno_val_idx.csv')
test_oc['RecordID'].to_csv('./data/Pheno_test_idx.csv')

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
# maxv = max(ts['value'])
# print(maxv)
# print(len(ts))
# index = ts[(~ts.value.between(-2, 2))&(~ts.variable.isin(['Age', 'Gender']))].index
# print(len(index))
# ts.drop(index, inplace=True)
# print(len(ts))
# maxv = max(ts['value'])
# print(maxv)
# print(np.percentile(ts['value'], 99))
pickle.dump([ts, oc, train_ind, valid_ind, test_ind], open('mimic_iii_pheno.pkl','wb'))
