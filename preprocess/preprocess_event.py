import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from data.dataset import Physionet2012DataReader
from sklearn.model_selection import train_test_split
import re
import json

def refill_nan(x_, save_path = None, load_path = None):
    x = x_.copy()
    gen = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    if load_path != None:
        data = np.load(load_path)
        p5 = data['p5']
        p95 = data['p95']
        means = data['means']
    else:
        p5 = np.zeros(gen.shape[0])
        p95 = np.zeros(gen.shape[0])
        means = np.zeros(gen.shape[0])
        for i in range(gen.shape[0]):
            datai = gen[i][~np.isnan(gen[i])]
            if len(datai[np.nonzero(datai)]) == 0:
                print('zero modality, skipped')
                continue
            p5[i] = np.percentile(datai[np.nonzero(datai)], 5)
            p95[i] = np.percentile(datai[np.nonzero(datai)], 95)
            means[i] = np.percentile(datai[np.nonzero(datai)], 50)
        if save_path != None:
            np.savez(save_path, p5 = p5, p95 = p95, means = means)
    # means = np.nanmean(x.transpose(1, 0, 2).reshape(x.shape[1], -1), axis = 1)
    for i in range(gen.shape[0]):
        if p5[i] == p95[i]:
            x[:, i, :] = 0
            continue
        x[:, i, :][np.nonzero(x[:, i, :])] = (x[:, i, :][np.nonzero(x[:, i, :])] - means[i]) / (p95[i] - p5[i])
    return x

def extract_mimiciii(path, max_ts_len, max_event_len, suffix, fts1, fts2):
    # extract train
    train_path = os.path.join(path, suffix)
    # files = 'bg,uo,gcs,lab'.split(',')
    fts = fts1 + fts2

    total_samples = len(os.listdir(train_path))

    vital_masks = np.zeros((total_samples, max_ts_len))
    vitals = np.zeros((total_samples, 37, max_ts_len))
    motality = np.zeros((total_samples))
    length_of_stay = np.zeros((total_samples))
    statics = np.zeros((total_samples, 3))
    cnt = 0
    ids = np.zeros(len(os.listdir(train_path)))

    ev_cache = np.zeros((len(fts), 150))
    seq_cache = np.zeros((ev_cache.shape[0]*ev_cache.shape[1], 2))
    template = np.outer(np.arange(ev_cache.shape[0])+1, np.ones(ev_cache.shape[1])).reshape(-1)

    event_times = np.zeros((total_samples, 400))
    event_types = np.zeros((total_samples, 400))

    for patient in tqdm(os.listdir(train_path)):
        pathi = os.path.join(train_path, patient)
        staypath = os.path.join(pathi, 'stays.csv')
        stays = pd.read_csv(staypath)
        for epis in range(1, 10):
            vitalpath = os.path.join(pathi, 'episode{}_vital_timeseries.csv'.format(epis))
            if not os.path.exists(vitalpath): # no vital file recorded
                break

            los = stays['LOS'].iloc[epis-1]*24
            if los<48:
                continue
            vital = pd.read_csv(vitalpath)
            vital = vital[vital['Hours'] < 48]
            if vital.shape[0] < 10:
                continue
            e1_path = os.path.join(pathi, 'episode{}_full_events_timeseries.csv'.format(epis))
            event1= pd.read_csv(e1_path)
            e2_path = os.path.join(pathi, 'episode{}_full_events2_timeseries.csv'.format(epis))
            event2 = pd.read_csv(e2_path)

            # event1 = event1[event1['Hours']<47]
            # event2 = event2[event2['Hours']<47]
            event1 = event1.set_index('Hours').loc[:, fts1].dropna(how = 'all').reset_index()
            event2 = event2.set_index('Hours').loc[:, fts2].dropna(how = 'all').reset_index()
            # event1 = event1[event1['Hours']>=0]
            # event2 = event2[event2['Hours']>=0]
            event1['Hours'] = pd.to_timedelta(event1['Hours'], unit = 'h')
            event2['Hours'] = pd.to_timedelta(event2['Hours'], unit = 'h')

            merged = pd.merge(event1, event2, how = 'outer', on = 'Hours').sort_values('Hours')
            # print(merged['Hours'].to_list)
            vital['Hours'] = pd.to_timedelta(vital['Hours'], unit = 'h')
            # vital = vital.set_index('Hours').resample('1h').mean().reset_index()
            # vital = vital.set_index('Hours').drop(columns=['SPO2']).dropna(how = 'all').reset_index()
            merged = pd.merge_asof(left = vital, right = merged, on = 'Hours',  direction= 'nearest', tolerance= pd.Timedelta(30, 'm')).sort_values('Hours')
            # merged = pd.merge(left = vital, right = merged, how = 'outer', on = 'Hours')
            merged = merged.set_index('Hours').resample('1h').mean().reset_index()
            ft = merged.iloc[:, 1:].to_numpy().transpose(1, 0)
            end = min(ft.shape[1], max_ts_len)
            vitals[cnt, :, :end] = ft[:, :end]
            vital_masks[cnt, :end] = 1

            event1 = event1.set_index('Hours')
            for ft in fts1:
                df = event1.loc[:, ft].dropna()
                length = min(len(df), 150)
                ev_cache[fts.index(ft), :length] = (df.index / np.timedelta64(1, 'h')).to_numpy()[:length]

            event2 = event2.set_index('Hours')
            for ft in fts2:
                df = event2.loc[:, ft].dropna()
                length = min(len(df), 150)
                ev_cache[fts.index(ft), :length] = (df.index / np.timedelta64(1, 'h')).to_numpy()[:length]

            seq_cache[:, 0] = ev_cache.reshape(-1)
            seq_cache[:, 1] = template
            seq_cache[:, 1][seq_cache[:, 0] == 999] = 0
            seq_cache = seq_cache[np.argsort(seq_cache[:, 0])]
            seq_cache[:, 0][seq_cache[:, 0] == 999] = 0
            seq_cache[:, 0] -= seq_cache[0, 0]
            event_times[cnt] = seq_cache[:max_event_len, 0]
            event_types[cnt] = seq_cache[:max_event_len, 1]

            motality[cnt] = stays.loc[:, 'MORTALITY_INHOSPITAL'].iloc[epis-1]
            length_of_stay[cnt] = stays.loc[:, 'LOS'].iloc[epis-1]
            # statics[cnt] = stays.loc[:, sts].head(1).to_numpy()
            if stays['GENDER'].iloc[0] == 'F':
                statics[cnt][0] = 1
            else:
                statics[cnt][1] = 0
            statics[cnt][2] = stays['AGE'].iloc[0]

            ids[cnt] = int(patient)*10 + epis
            cnt += 1

    vitals[np.isnan(vitals)] = 0
    vitals = refill_nan(vitals, save_path='./data/mimiciii_statics.npz')

    # length_of_stay = np.digitize(length_of_stay- 2, bins = [1,2,3,4,5,6,7,8,14])

    if suffix == 'test':
        # vitals = refill_nan(vitals, load_path='./data/mimiciii_statics.npz')
        np.savez('./data/physionet/mimiciii_refill_{}.npz'.format(suffix),
        event_times = event_times[:cnt], event_types = event_types[:cnt], ids = ids[:cnt],
        vitals = vitals[:cnt], vital_masks = vital_masks[:cnt], mortality = motality[:cnt], statics = statics[:cnt], los = length_of_stay[:cnt])
    else:
        # vitals = refill_nan(vitals, save_path='./data/mimiciii_statics.npz')
        # np.savez(''.format(suffix),
        # event_times = event_times[:cnt], event_values = event_values[:cnt], event_masks = padding_masks[:cnt], ids = ids[:cnt],
        # vitals = vitals[:cnt], vital_masks = vital_masks[:cnt], mortality = motality[:cnt], statics = statics[:cnt])
        train_ids, val_ids = train_test_split(ids[:cnt], test_size=0.2, stratify=motality[:cnt], random_state=13)
        train_ids = np.isin(ids[:cnt], train_ids)
        val_ids = np.isin(ids[:cnt], val_ids)
        np.savez('./data/physionet/mimiciii_refill_{}.npz'.format('train'),
        event_times = event_times[:cnt][train_ids], event_types = event_types[:cnt][train_ids], ids = ids[:cnt][train_ids], los = length_of_stay[:cnt][train_ids],
        vitals = vitals[:cnt][train_ids], vital_masks = vital_masks[:cnt][train_ids], mortality = motality[:cnt][train_ids], statics = statics[:cnt][train_ids])
        np.savez('./data/physionet/mimiciii_refill_{}.npz'.format('val'),
        event_times = event_times[:cnt][val_ids], event_types = event_types[:cnt][val_ids], ids = ids[:cnt][val_ids], los = length_of_stay[:cnt][val_ids],
        vitals = vitals[:cnt][val_ids], vital_masks = vital_masks[:cnt][val_ids], mortality = motality[:cnt][val_ids], statics = statics[:cnt][val_ids])
    
    # extract test

def extract_physionet(reader, max_ts_len = 48, max_event_len = 400):
    # extract train
    # pass
    vitals = np.zeros((len(reader), len(reader.vital_features), max_ts_len))
    vital_masks = np.zeros((len(reader), max_ts_len))
    # event_times = np.zeros((len(reader), len(reader.event_features), maxlen))
    # event_values = np.zeros((len(reader), len(reader.event_features), maxlen))
    event_times = np.zeros((len(reader), max_event_len))
    event_types = np.zeros((len(reader), max_event_len))
    ev_cache = np.zeros((len(reader.event_features), 150))
    seq_cache = np.zeros((ev_cache.shape[0]*ev_cache.shape[1], 2))
    template = np.outer(np.arange(ev_cache.shape[0])+1, np.ones(ev_cache.shape[1])).reshape(-1)
    mortality = np.zeros(len(reader))
    statics = np.zeros((len(reader), len(reader.expanded_static_features)))
    ids = np.zeros(len(reader))
    cnt = 0
    fts = reader.event_features
    for id, static, vital, event, target in tqdm(reader):
        # vital = vital.fillna(method = 'ffill')
        # vital = vital.resample('1h').mean().fillna(method = 'ffill').fillna(method = 'bfill')
        vital = vital.resample('1h').mean()
        vital = vital.to_numpy().transpose((1, 0))
        end = min(vital.shape[1], max_ts_len)
        vitals[cnt, :, :end] = vital[:, :end]
        vital_masks[cnt, :end] = 1

        ev_cache.fill(999)
        seq_cache.fill(0)

        for ft in event.keys():
            df = event.loc[:, ft].dropna()
            length = min(len(df), 150)
            ev_cache[fts.index(ft), :length] = (df.index / np.timedelta64(1, 'h')).to_numpy()[:length]
        seq_cache[:, 0] = ev_cache.reshape(-1)
        seq_cache[:, 1] = template
        seq_cache[:, 1][seq_cache[:, 0] == 999] = 0
        seq_cache = seq_cache[np.argsort(seq_cache[:, 0])]
        seq_cache[:, 0][seq_cache[:, 0] == 999] = 0
        seq_cache[:, 0] -= seq_cache[0, 0]
        event_times[cnt] = seq_cache[:max_event_len, 0]
        event_types[cnt] = seq_cache[:max_event_len, 1]
        
        mortality[cnt] = target['In-hospital_death']
        statics[cnt] = static.to_numpy()
        ids[cnt] = id
        cnt += 1

    return vitals[:cnt], vital_masks[:cnt], event_times[:cnt], event_types[:cnt], mortality[:cnt], statics[:cnt], ids[:cnt]

def extract_physionet_all(id_lists, vital_ft, event_ft):
    av, avm, aet, aev, am, ast, aid = extract_physionet(Physionet2012DataReader(['./data/physionet/set-a/'], './data/physionet/Outcomes-a.txt', vital_ft, event_ft))
    bv, bvm, bet, bev, bm, bst, bid = extract_physionet(Physionet2012DataReader(['./data/physionet/set-b/'], './data/physionet/Outcomes-b.txt', vital_ft, event_ft))
    cv, cvm, cet, cev, cm, cst, cid = extract_physionet(Physionet2012DataReader(['./data/physionet/set-c/'], './data/physionet/Outcomes-c.txt', vital_ft, event_ft))
    vitals = np.concatenate((av, bv, cv), axis = 0)
    vital_masks = np.concatenate((avm, bvm, cvm), axis = 0)
    event_times = np.concatenate((aet, bet, cet), axis = 0)
    event_types = np.concatenate((aev, bev, cev), axis = 0)
    mortality = np.concatenate((am, bm, cm), axis = 0)
    sts = np.concatenate((ast, bst, cst), axis = 0)
    ids = np.concatenate((aid, bid, cid), axis = 0)

    suffixes = ['train', 'val', 'test']
    for i in range(3):
        id_list = id_lists[i]
        idx = np.isin(ids, id_list)
        vital_i = vitals[idx]
        print(vital_i.shape)

        vital_i[np.isnan(vital_i)] = 0
        if i == 0:
            vital_i = refill_nan(vital_i, save_path= './data/physionet/p12_statics.npz')
        else:
            vital_i = refill_nan(vital_i, load_path= './data/physionet/p12_statics.npz')

        statics_i = sts[idx]
        statics_i[:, -1][statics_i[:, -1] == -1] = np.mean(statics_i[:, -1][statics_i[:, -1] != -1])

        np.savez('./data/physionet/physionet12_full_{}.npz'.format(suffixes[i]),
        event_times = event_times[idx], event_types = event_types[idx], ids = ids[idx],
        vitals = vital_i, vital_masks = vital_masks[idx], mortality = mortality[idx], statics = statics_i)

def extract_pheno(path, max_ts_len, max_event_len, suffix, fts1, fts2):
    train_path = os.path.join(path, suffix)
    # files = 'bg,uo,gcs,lab'.split(',')
    fts = fts1 + fts2
    vital_fts = json.load(open('./data/mimic/vital_fts.json', 'r'))
    event_times = np.zeros((len(os.listdir(train_path)), len(fts1) + len(fts2), max_event_len))
    event_values = np.zeros((len(os.listdir(train_path)), len(fts1) + len(fts2), max_event_len))
    vital_masks = np.zeros((len(os.listdir(train_path)), max_ts_len))
    vitals = np.zeros((len(os.listdir(train_path)), 36, max_ts_len))
    pheno = np.zeros((len(os.listdir(train_path)), 25))
    statics = np.zeros((len(os.listdir(train_path)), 3))
    cnt = 0
    ids = np.zeros(len(os.listdir(train_path)))
    listfile = pd.read_csv(os.path.join(train_path, 'listfile.csv'))
    for vitalpath in tqdm(listfile['stay'].to_list()):
        id = vitalpath.split('_')[0]
        epis = vitalpath.split('_')[1].split('e')[-1]
        pathi = os.path.join(train_path, )
        
        vital = pd.read_csv(os.path.join(train_path, vitalpath))
        # if vital.shape[0] < 5:
        #     continue
        vital = vital.loc[:, ['Hours'] + vital_fts]
        e1_path = os.path.join(pathi, '{}_episode{}_full_events_timeseries.csv'.format(id, epis))
        e2_path = os.path.join(pathi, '{}_episode{}_full_events2_timeseries.csv'.format(id, epis))
        if not os.path.exists(e1_path) or not os.path.exists(e2_path):
            print(vitalpath)
            continue
        event1= pd.read_csv(e1_path)
        event2 = pd.read_csv(e2_path)

        event1 = event1.set_index('Hours').loc[:, fts1].dropna(how = 'all').reset_index()
        event2 = event2.set_index('Hours').loc[:, fts2].dropna(how = 'all').reset_index()

        merged = pd.merge(event1, event2, how = 'outer', on = 'Hours').sort_values('Hours')
        merged['Hours'] = pd.to_timedelta(merged['Hours'], unit = 'h')
        vital['Hours'] = pd.to_timedelta(vital['Hours'], unit = 'h')
        merged = pd.merge_asof(left = vital, right = merged, on = 'Hours', direction= 'nearest', tolerance= pd.Timedelta(30, 'm'))

        ft = merged.iloc[:, 1:].to_numpy().transpose(1, 0)
        end = min(ft.shape[1], max_ts_len)
        vitals[cnt, :, :end] = ft[:, :end]
        vital_masks[cnt, :end] = 1

        for ft in fts1:
            df = event1.loc[:, ['Hours', ft]]
            df = df[~df[ft].isnull()]
            length = min(len(df), max_event_len)
            event_times[cnt, fts.index(ft), :length] = df.loc[:, 'Hours'].head(length).to_numpy()
            event_values[cnt, fts.index(ft), :length] = df.loc[:, ft].head(length).to_numpy()

        for ft in fts2:
            df = event2.loc[:, ['Hours', ft]]
            df = df[~df[ft].isnull()]
            length = min(len(df), max_event_len)
            event_times[cnt, fts.index(ft), :length] = df.loc[:, 'Hours'].head(length).to_numpy()
            event_values[cnt, fts.index(ft), :length] = df.loc[:, ft].head(length).to_numpy()

        ids[cnt] = int(id)*10 + int(epis)
        col = listfile[listfile['stay'] == vitalpath]
        pheno[cnt] = col.iloc[:, 2:].to_numpy()
        cnt+=1
    print(cnt)
    vitals[np.isnan(vitals)] = 0

    if suffix == 'test':
        vitals = refill_nan(vitals, load_path='')
        np.savez('./data/mimiciii_pheno_{}.npz'.format(suffix),
        event_times = event_times[:cnt], event_values = event_values[:cnt], ids = ids[:cnt],
        vitals = vitals[:cnt], vital_masks = vital_masks[:cnt], pheno_labels = pheno[:cnt], statics = statics[:cnt])
    else:
        vitals = refill_nan(vitals, save_path='')
        train_ids, val_ids = train_test_split(ids[:cnt], test_size=0.2, random_state=13)
        train_ids = np.isin(ids[:cnt], train_ids)
        val_ids = np.isin(ids[:cnt], val_ids)
        np.savez('',
        event_times = event_times[:cnt][train_ids], event_values = event_values[:cnt][train_ids], ids = ids[:cnt][train_ids],  
        vitals = vitals[:cnt][train_ids], vital_masks = vital_masks[:cnt][train_ids], pheno_labels = pheno[:cnt][train_ids], statics = statics[:cnt][train_ids])
        np.savez('',
        event_times = event_times[:cnt][val_ids], event_values = event_values[:cnt][val_ids], ids = ids[:cnt][val_ids], 
        vitals = vitals[:cnt][val_ids], vital_masks = vital_masks[:cnt][val_ids], pheno_labels = pheno[:cnt][val_ids], statics = statics[:cnt][val_ids])

def extract_los(path, max_ts_len, max_event_len, suffix, fts1, fts2):
    train_path = os.path.join(path, suffix)
    # files = 'bg,uo,gcs,lab'.split(',')
    fts = fts1 + fts2
    vital_fts = json.load(open('./data/mimic/vital_fts.json', 'r'))
    
    listfile = pd.read_csv(os.path.join(train_path, 'listfile.csv'))
    listfile = listfile.loc[~(listfile['stay'].isin(['7805_episode2_timeseries.csv', '27394_episode1_timeseries.csv']))]
    ids = listfile['stay'].str.split('_', n = 2, expand=True).iloc[:, 0]
    epis = listfile['stay'].str.split('_', n = 3, expand=True).iloc[:, 1]
    epis = epis.str.split('e', expand=True).iloc[:, -1]
    epis = pd.to_numeric(epis).to_numpy()
    ids = pd.to_numeric(ids).to_numpy()
    # print(id.shape, epis.shape, len(listfile['period_length']), len(listfile['y_true']))
    y_trues = np.digitize(listfile['y_true'].to_numpy(), bins = [1,2,3,4,5,6,7,8,14])

    if suffix == 'train':
        idx_low = np.where(y_trues<8)[0]
        idx8 = np.where(y_trues == 8)[0]
        idx9 = np.where(y_trues == 9)[0]
        idx8 = np.random.choice(idx8, int(idx_low.shape[0] / 8), replace=False)
        idx9 = np.random.choice(idx9, int(idx_low.shape[0] / 8), replace=False)
        idx = np.concatenate((idx_low, idx8, idx9), axis = 0)

        ids = ids[idx]
        epis = epis[idx]
        period_lengths = listfile['period_length'].to_numpy()[idx]
        y_trues = y_trues[idx]
    else:
        period_lengths = listfile['period_length'].to_numpy()
    print(np.bincount(y_trues))

    ev_cache = np.zeros((len(fts), 150))
    seq_cache = np.zeros((ev_cache.shape[0]*ev_cache.shape[1], 2))
    template = np.outer(np.arange(ev_cache.shape[0])+1, np.ones(ev_cache.shape[1])).reshape(-1)

    for i in tqdm(range(len(y_trues))):
        y_true = y_trues[i]
        period_length = period_lengths[i]
        
        id = ids[i]
        epi = epis[i]

        vital_path = os.path.join(train_path, '{}_episode{}_timeseries.csv'.format(id, epi))
        e1_path = os.path.join(train_path, '{}_episode{}_full_events_timeseries.csv'.format(id, epi))
        e2_path = os.path.join(train_path, '{}_episode{}_full_events2_timeseries.csv'.format(id, epi))
        if not os.path.exists(vital_path) or not os.path.exists(e1_path) or not os.path.exists(e2_path):
            print(id, epi)
            continue
        vital = pd.read_csv(vital_path)
        event1= pd.read_csv(e1_path)
        event2 = pd.read_csv(e2_path)
        # event1 = event1.set_index('Hours').loc[:, fts1].dropna(how = 'all').reset_index()
        # event2 = event2.set_index('Hours').loc[:, fts2].dropna(how = 'all').reset_index()

        vital = vital.loc[:, ['Hours'] + vital_fts].dropna(how = 'all')
        vital['Hours'] = pd.to_timedelta(vital['Hours'])
        # ft = merged.iloc[:, 1:].to_numpy().transpose(1, 0)

        vital = vital[vital['Hours'] < pd.Timedelta(period_length, unit = 'h')] # vitals
        vital = vital[vital['Hours'] > pd.Timedelta(period_length - 48, unit = 'h')]
        ev1 = event1[event1['Hours'] < period_length]
        ev1 = ev1[ev1['Hours'] > period_length-48]
        ev2 = event2[event2['Hours'] < period_length]
        ev2 = ev2[ev2['Hours'] > period_length-48]

        ev1 = ev1.set_index('Hours').loc[:, fts1].dropna(how = 'all').reset_index()
        ev2 = ev2.set_index('Hours').loc[:, fts2].dropna(how = 'all').reset_index()

        merged = pd.merge(ev1, ev2, how = 'outer', on = 'Hours').sort_values('Hours')
        merged['Hours'] = pd.to_timedelta(merged['Hours'], unit = 'h')
        vital = vital.set_index('Hours').resample('1h').mean().reset_index()
        vital = pd.merge_asof(left = vital, right = merged, on = 'Hours', direction= 'nearest', tolerance= pd.Timedelta(30, 'm'))

        end = min(vital.shape[1], max_ts_len)
        vital_mask = np.zeros(max_ts_len)
        vital_mask[:end] = 1
        # vitals[i, :, :end] = fti[:, :end]
        # padding_masks[i, :end] = 1
        ev_cache.fill(999)
        seq_cache.fill(0)

        for ft in fts1:
            df = ev1.set_index('Hours').loc[:, ft].dropna()
            length = min(len(df), 50)
            ev_cache[fts.index(ft), :length] = (df.index).to_numpy()[:length]

        for ft in fts2:
            df = ev2.set_index('Hours').loc[:, ft].dropna()
            length = min(len(df), 50)
            ev_cache[fts.index(ft), :length] = (df.index).to_numpy()[:length]
        
        seq_cache[:, 0] = ev_cache.reshape(-1)
        seq_cache[:, 1] = template
        seq_cache[:, 1][seq_cache[:, 0] == 999] = 0
        seq_cache = seq_cache[np.argsort(seq_cache[:, 0])]
        seq_cache[:, 0][seq_cache[:, 0] == 999] = 0
        seq_cache[:, 0] -= seq_cache[0, 0]

        savepath = ''.format(suffix, int(id), int(epi), int(period_length))
        np.savez(savepath, vital = vital, vital_mask = vital_mask, event_times = seq_cache[:max_event_len, 0], event_types = seq_cache[:max_event_len, 1], y_true = y_true)

def preprocess_los(suffix = 'train'):
    root = ''
    path = os.path.join(root, suffix)
    listfile = pd.read_csv(os.path.join(root, '../length-of-stay/{}'.format(suffix), 'listfile.csv'))
    listfile = listfile[listfile['stay']!='7805_episode2_timeseries.csv']
    
    statics = np.load('./data/mimiciii_statics.npz')
    p5, p95, mean = statics['p5'], statics['p95'], statics['means']
    p5 = np.delete(p5, 5).reshape((-1, 1))
    p95 = np.delete(p95, 5).reshape((-1, 1))
    mean = np.delete(mean, 5).reshape((-1, 1))
    for i in tqdm(range(len(listfile))):
        entry = listfile.iloc[i]
        stay = entry['stay']
        id = stay.split('_')[0]
        epis = stay.split('_')[1].split('e')[-1]

        vital_path = os.path.join(path, '{}_episode{}_len{}_vitals.npy'.format(int(id), int(epis), int(entry['period_length'])))
        vital = np.load(vital_path)
        vital = (vital - p5) / (p95 - p5)
        vital[np.isnan(vital)] = 0
        # print(vital[5])
        # break
        np.save(vital_path, vital)

def extract_entry(path = ''):
    entry_path = os.path.join(path,'mimic2physionet.csv')
    entry = pd.read_csv(entry_path)
    # table_events,column_events,feature_physionet,confidence
    fts = entry.loc[entry['confidence'].isin([1, 2])]
    column_1 = fts.loc[fts['table_events'] == 'full_events_timeseries']
    column_2 = fts.loc[fts['table_events'] == 'full_events2_timeseries']
    # mimic_param = zip(ft_con2['table_events'].tolist(), ft_con2['column_events'].tolist())
    fts1 = column_1['column_events'].tolist()
    fts2 = column_2['column_events'].tolist()
    sts = ['']
    # for item in mimic_param
    print(fts['column_events'].tolist())
    extract_mimiciii(path, 48, 400, 'train', fts1, fts2)
    extract_mimiciii(path, 48, 400, 'test', fts1, fts2)
    # extract_pheno(os.path.join(path, 'phenotyping'), 150, 100, 'train', fts1, fts2)
    # extract_pheno(os.path.join(path, 'phenotyping'), 150, 100, 'test', fts1, fts2)
    # extract_los(os.path.join(path, 'length-of-stay'), 150, 100, 'train', fts1, fts2)
    # extract_los(os.path.join(path, 'length-of-stay'), 150, 100, 'test', fts1, fts2)
    # extract_los('../tmp', 48, 400, 'train', fts1, fts2)

    # p_vitals = ['DiasABP', 'Glucose', 'HR', 'MAP', 'RespRate', 'SysABP', 'Temp']
    # p_fts1 = column_1['feature_physionet'].tolist()
    # p_fts2 = column_2['feature_physionet'].tolist()
    # p_fts1[p_fts1.index('TropI')] = 'TroponinI'
    # p_fts1[p_fts1.index('TropT')] = 'TroponinT'

    # ids = np.load('./data/physionet/apc_id.npz', allow_pickle=True)
    # train_ids = [int(i) for i in ids['train']]
    # test_ids = [int(i) for i in ids['test']]
    # val_ids = [int(i) for i in ids['val']]
    # train_ids = pd.read_csv('~/codes/medical_ts_datasets/medical_ts_datasets/resources/physionet2012/train_listfile.csv')['RecordID'].to_list()
    # train_ids = [int(i) for i in train_ids]
    # test_ids = pd.read_csv('~/codes/medical_ts_datasets/medical_ts_datasets/resources/physionet2012/test_listfile.csv')['RecordID'].to_list()
    # test_ids = [int(i) for i in test_ids]
    # val_ids = pd.read_csv('~/codes/medical_ts_datasets/medical_ts_datasets/resources/physionet2012/val_listfile.csv')['RecordID'].to_list()
    # val_ids = [int(i) for i in val_ids]
    # extract_physionet_all([train_ids, val_ids, test_ids], p_vitals + p_fts1 + p_fts2, p_fts1 + p_fts2)
    # extract_physionet_all('test', p_vitals + p_fts1 + p_fts2, p_fts1 + p_fts2)

def extract_amsterdam(path = ''):
    entry = pd.read_csv('')
    fts = entry.loc[entry['confidence'].isin([1, 2])]
    vital_fts = json.load(open('./data/mimic/vital_fts.json', 'r'))
    column_1 = fts.loc[fts['table_events'] == 'full_events_timeseries']
    column_2 = fts.loc[fts['table_events'] == 'full_events2_timeseries']
    fts1 = column_1['column_events'].tolist()
    fts2 = column_2['column_events'].tolist()
    fts = fts1 + fts2
    vital_fts = (vital_fts + fts).copy()
    print(vital_fts.index('Glascow coma scale total'))
    vital_fts.remove('Glascow coma scale total')
    print(len(vital_fts), len(fts))

    total_samples = len(os.listdir(path))
    vitals = np.zeros((total_samples, 36, 48))
    vital_masks = np.zeros((total_samples, 48))
    
    ev_cache = np.zeros((len(fts), 150))
    seq_cache = np.zeros((ev_cache.shape[0]*ev_cache.shape[1], 2))
    template = np.outer(np.arange(ev_cache.shape[0])+1, np.ones(ev_cache.shape[1])).reshape(-1)

    event_times = np.zeros((total_samples, 400))
    event_types = np.zeros((total_samples, 400))
    statics = np.zeros((total_samples, 2))
    mortality = np.zeros(total_samples)
    los = np.zeros(total_samples)

    cnt = 0
    admissions = pd.read_csv(os.path.join(path, '../admission.csv'))

    record_idx = []
    idx = [[], [], []]
    for suffix in ['train', 'val', 'test']:
        record_idx.append(pd.read_csv('./data/AUMC_{}_idx.csv'.format(suffix))['RecordID'].to_list())
    for usr in tqdm(os.listdir(path)):
        if usr == 'Amsterdam-Mimic.xlsx':
            continue
        entries = admissions[admissions['patientid']==int(usr)]
        
        for i in range(len(entries)):
            entry = entries[entries['admissioncount']==i+1]
            if entry.iloc[0]['lengthofstay']<48:
                continue
            # morts.append(int(entry['destination']=='Overleden'))

            df = pd.read_csv(os.path.join(path, usr, 'episode{}_full_events_timeseries.csv'.format(i+1)))
            if i > 1:
                df['Hours'] -= df['Hours'].min()
            else:
                df = df[df['Hours']>0]
            df = df[df['Hours']<48].sort_values(by =['Hours'])
            df['Hours'] = pd.to_timedelta(df['Hours'], unit = 'h')
            df = df.set_index('Hours')

            # TODO: fix possible time issue
            # print(df.resample('h').mean().loc[:, vital_fts])
            # break
            vital = df.resample('h').mean().loc[:, vital_fts].to_numpy().transpose(1, 0)
            end = min(vital.shape[1], 48)
            vitals[cnt, :16, :end] = vital[:16, :end]
            vitals[cnt, 17:, :end] = vital[16:, :end]
            vital_masks[cnt, :end] = 1

            ev_cache.fill(999)
            seq_cache.fill(0)
            for ft in fts:
                if ft == 'Glascow coma scale total':
                    continue
                dfi = df.loc[:, ft].dropna()
                length = min(len(dfi), 150)
                ev_cache[fts.index(ft), :length] = (dfi.index / np.timedelta64(1, 'h')).to_numpy()[:length]
            seq_cache[:, 0] = ev_cache.reshape(-1)
            seq_cache[:, 1] = template
            seq_cache[:, 1][seq_cache[:, 0] == 999] = 0
            seq_cache = seq_cache[np.argsort(seq_cache[:, 0])]
            seq_cache[:, 0][seq_cache[:, 0] == 999] = 0
            seq_cache[:, 0] -= seq_cache[0, 0]
            event_times[cnt] = seq_cache[:400, 0]
            event_types[cnt] = seq_cache[:400, 1]

            age = str(entry.iloc[0]['agegroup'])
            age = float(re.search(r'[0-9]+', age).group())

            gender = float(entry.iloc[0]['gender'] == 'Man')

            if np.isnan(age) or np.isnan(gender):
                print('Error')
                print(str(entry.iloc[0]['agegroup']), entry.iloc[0]['gender'], entry.iloc[0]['heightgroup'])
                exit()

            statics[cnt] = [age, gender]
            mort = int(entry['destination']=='Overleden')
            mortality[cnt] = mort
            los[cnt] = entry['lengthofstay'] / 24

            recordid = '{}#{}'.format(int(usr), i)
            for i in range(3):
                if recordid in record_idx[i]:
                    idx[i].append(cnt)
                    break
            cnt += 1
    print([len(i) for i in idx])
    # print(cnt, vitals.shape[0])
    vitals[np.isnan(vitals)] = 0
    vitals = refill_nan(vitals, save_path='./data/physionet/AUMC_statics.npz')

    suffixes = ['train', 'val', 'test']
    for i in range(3):
        np.savez('./data/physionet/aumc_{}.npz'.format(suffixes[i]), vitals = vitals[:cnt][idx[i]], vital_masks = vital_masks[:cnt][idx[i]], los = los[:cnt][idx[i]],
                event_times = event_times[:cnt][idx[i]], event_types = event_types[:cnt][idx[i]], mortality = mortality[:cnt][idx[i]], statics = statics[:cnt][idx[i]])

if __name__ == "__main__":
    # extract_mimiciii(suffix='train')
    # extract_physionet_all('test')
    # extract_entry()
    extract_amsterdam()