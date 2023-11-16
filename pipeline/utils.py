import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pandas as pd

class InfIter:
    def __init__(self,_loader):
        self._loader = _loader
        self._iter = iter(_loader)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)


class MaskedMSELoss(torch.nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self):

        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        # print(masked_pred.shape, masked_true.shape)
        return self.mse_loss(masked_pred, masked_true)

class NaiveDataset(object):
    def __init__(self, tensors, targets):
        assert tensors.shape[0] == targets.shape[0]
        self.tensors = tensors
        self.targets = targets
    def __getitem__(self, i):
        return self.tensors[i], self.targets[i]
    def __len__(self):
        return self.tensors.shape[0]

class EventDataset(object):
    def __init__(self, times, types, targets, statics):
        assert times.shape[0] == targets.shape[0] == types.shape[0]
        self.times = times
        self.types = types
        self.targets = targets
        self.statics = statics
    def __getitem__(self, i):
        return self.times[i], self.types[i], self.targets[i], self.statics[i]
    def __len__(self):
        return self.times.shape[0]

class MaskedDataset(object):
    def __init__(self, tensors, targets, masks, statics = None, have_statics = False):
        assert tensors.shape[0] == targets.shape[0] == masks.shape[0]
        self.tensors = tensors
        self.targets = targets
        self.masks = np.ma.make_mask(masks)
        self.statics = statics
        self.have_statics = have_statics
    def __getitem__(self, i):
        if self.have_statics:
            return self.tensors[i], self.targets[i], self.masks[i], self.statics[i]
        else:
            return self.tensors[i], self.targets[i], self.masks[i]
    def __len__(self):
        return self.tensors.shape[0]

class UnsupervisedMaskedDataset(object):
    def __init__(self, tensors, masks):
        self.tensors = tensors
        self.masks = np.ma.make_mask(masks)
    def __getitem__(self, i):
        return self.tensors[i], self.masks[i]
    def __len__(self):
        return self.tensors.shape[0]

class MixedDataset(object):
    def __init__(self, vitals, masks, times, types, targets, statics = None, has_statics = False):
        assert vitals.shape[0] == masks.shape[0] == times.shape[0] == types.shape[0] == targets.shape[0]
        self.vitals = vitals
        self.masks = np.ma.make_mask(masks)
        self.times = times
        self.types = types
        self.targets = targets
        self.statics = statics
        self.has_statics = has_statics
    def __getitem__(self, i):
        if self.has_statics:
            return self.vitals[i], self.masks[i], self.times[i], self.types[i], self.targets[i], self.statics[i]
        else:
            return self.vitals[i], self.masks[i], self.times[i], self.types[i], self.targets[i]
    def __len__(self):
        return self.times.shape[0]

class TruncDataset(object):
    def __init__(self, tensors, targets, masks, trunc_size = 7, statics = None, have_statics = False):
        assert tensors.shape[0] == targets.shape[0] == masks.shape[0]
        self.tensors = tensors
        self.targets = targets
        self.masks = np.ma.make_mask(masks)
        self.statics = statics
        self.have_statics = have_statics
        self.trunc_size = trunc_size
    def __getitem__(self, i):
        if self.have_statics:
            return self.tensors[i], self.tensors[i, :self.trunc_size], self.targets[i], self.masks[i], self.statics[i]
        else:
            return self.tensors[i], self.tensors[i, :self.trunc_size], self.targets[i], self.masks[i]
    def __len__(self):
        return self.tensors.shape[0]
    
class LOSDataset(object):
    def __init__(self, id, epis, los, y_true, path, ts_shape):
        self.id = id
        self.epis = epis
        self.los = los
        self.y_true = y_true
        self.path = path
        self.ts_shape = ts_shape
    def __getitem__(self, i):
        vital = np.zeros(self.ts_shape)
        mask = np.zeros(self.ts_shape[1])
        _vital = np.load(os.path.join(self.path, '{}_episode{}_len{}_vitals.npy'.format(int(self.id[i]), int(self.epis[i]), int(self.los[i]))))
        if _vital.shape[1] == 0:
            return self.__getitem__(i+1)
        leng = min(_vital.shape[1], self.ts_shape[1])
        vital[:, :leng] = _vital
        mask[:leng] = 1
        types = np.load(os.path.join(self.path, '{}_episode{}_len{}_event_types.npy'.format(int(self.id[i]), int(self.epis[i]), int(self.los[i]))))
        times = np.load(os.path.join(self.path, '{}_episode{}_len{}_event_times.npy'.format(int(self.id[i]), int(self.epis[i]), int(self.los[i]))))
        return vital, np.ma.make_mask(mask), times, types, self.y_true[i]
    def __len__(self):
        return self.id.shape[0]

def normalize(x):
    data_x = x.copy()
    x_ = np.transpose(data_x, (1, 2, 0)).reshape(data_x.shape[1], -1)
    p5 = np.percentile(x_, 5, axis = 1)
    p95 = np.percentile(x_, 95, axis = 1)
    data_x = (data_x - p5.reshape((-1, 1))) / (p95.reshape((-1, 1)) - p5.reshape((-1, 1)))
    return data_x

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
        x[:, i, :][np.nonzero(x[:, i, :])] = (x[:, i, :][np.nonzero(x[:, i, :])] - p5[i]) / (p95[i] - p5[i])
    return x

def decompose_events(times, unwrap = True):
    if unwrap:
        times = times['event_times']
    template = np.outer(np.arange(times.shape[1])+1, np.ones(times.shape[2])).reshape(-1)
    times = times.reshape(times.shape[0], times.shape[1]*times.shape[2])
    # print(times.shape)
    seq_sorted = np.zeros((times.shape[0], times.shape[1], 2))
    seq_sorted[:, :, 0] = times
    seq_sorted[:, :, 1] = template
    seq_sorted[:, :, 1][seq_sorted[:, :, 0] == 0] = 0
    seq_sorted[:, :, 1][seq_sorted[:, :, 0] >48] = 0
    seq_sorted[:, :, 1][seq_sorted[:, :, 0] <0] = 0
    seq_sorted[:, :, 0][seq_sorted[:, :, 1] == 0] = 999
    # for i in tqdm(range(seq_sorted.shape[0])):
    for i in range(seq_sorted.shape[0]):
        seq_sorted[i, :, :] = seq_sorted[i, :, :][np.argsort(seq_sorted[i, :, 0])]
    seq_sorted[:, :, 0] -= np.outer(np.min(seq_sorted[:, :, 0], axis = 1), np.ones(seq_sorted.shape[1]))
    seq_sorted[:, :, 0][seq_sorted[:, :, 1] == 0] = 0

    times, types = seq_sorted[:, :, 0], seq_sorted[:, :, 1]
    return times, types

def prepare_pheno(batch_size, max_event_len, ratio = 100):
    
    train_data = np.load('./data/physionet/mimiciii_pheno_train.npz')
    test_data = np.load('./data/physionet/mimiciii_pheno_test.npz')
            
    train_times, train_types = decompose_events(train_data)
    train_times = train_times[:, :max_event_len]
    train_types = train_types[:, :max_event_len]

    train_idx = (np.sum(train_types!=0, axis = 1) > 10)

    times, types = decompose_events(test_data)
    test_times = times[:, :max_event_len]
    test_types = types[:, :max_event_len]
    
    test_idx = (np.sum(test_types!=0, axis = 1) > 10)

    # times, types = decompose_events(val_data)
    # val_times = times[:, :max_event_len]
    # val_types = types[:, :max_event_len]
    # val_morts = val_data['mortality']
    
    # val_idx = (np.sum(val_types!=0, axis = 1) > 10)

    train_dataset = MixedDataset(train_data['vitals'], train_data['vital_masks'],
                        train_times, train_types, train_data['pheno_labels'], train_data['statics'], True)
    test_dataset = MixedDataset(test_data['vitals'], test_data['vital_masks'],
                        test_times, test_types, test_data['pheno_labels'], test_data['statics'], True)
    if ratio == 100:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=13)
        ts_shape = train_data['vitals'].shape
        static_shape = train_data['statics'].shape
        return ts_shape, static_shape, [DataLoader(dataset, batch_size= batch_size, shuffle=True) for dataset in [train_dataset, val_dataset, test_dataset]]
    else:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=13)
        pretrain_dataset, train_dataset = train_test_split(train_dataset, train_size = ratio / 100, random_state=13)
        ts_shape = train_data['vitals'].shape
        static_shape = train_data['statics'].shape
        return ts_shape, static_shape, [DataLoader(dataset, batch_size= batch_size, shuffle=True) for dataset in [pretrain_dataset, train_dataset, val_dataset, test_dataset]]

def prepare_balanced_loaders(dataname, batch_size, max_ts_len, max_event_len):
    if dataname == 'mimic':
        train_data = np.load('./data/physionet/mimiciii_train.npz')
        val_data = np.load('./data/physionet/mimiciii_val.npz')
        test_data = np.load('./data/physionet/mimiciii_test.npz')
        max_len = min(max_ts_len, train_data['vitals'].shape[2])
        train_vitals = np.concatenate((train_data['vitals'][:, :5, :max_len], train_data['vitals'][:, 6:, :max_len]), axis = 1)
        val_vitals = np.concatenate((val_data['vitals'][:, :5, :max_len], val_data['vitals'][:, 6:, :max_len]), axis = 1)
        test_vitals = np.concatenate((test_data['vitals'][:, :5, :max_len], test_data['vitals'][:, 6:, :max_len]), axis = 1)
    elif dataname == 'AUMC':
        train_data = np.load('./data/physionet/aumc_train.npz')
        val_data = np.load('./data/physionet/aumc_val.npz')
        test_data = np.load('./data/physionet/aumc_test.npz')
        max_len = min(max_ts_len, train_data['vitals'].shape[2])
        train_vitals = train_data['vitals'][:, :max_len]
        val_vitals = val_data['vitals'][:, :max_len]
        test_vitals = test_data['vitals'][:, :max_len]
    else:
        train_data = np.load('./data/physionet/physionet12_train.npz')
        val_data = np.load('./data/physionet/physionet12_val.npz')
        test_data = np.load('./data/physionet/physionet12_test.npz')
        max_len = min(max_ts_len, train_data['vitals'].shape[2])
        train_vitals = train_data['vitals'][:, :max_len]
        val_vitals = val_data['vitals'][:, :max_len]
        test_vitals = test_data['vitals'][:, :max_len]

    train_idx = (np.sum(train_data['event_types']>0, axis = 1) > 5)
    train_times = train_data['event_times'][train_idx]
    train_times -= np.min(train_times, axis = 1).reshape((-1, 1))
    train_types = train_data['event_types'][train_idx]
    train_morts = train_data['mortality'][train_idx]
    train_vitals = train_vitals[train_idx]
    train_vital_masks = train_data['vital_masks'][train_idx]
    train_statics = train_data['statics'][train_idx]
    print(min(np.sum(train_types, axis = 1)))

    test_idx = (np.sum(train_data['event_types'], axis = 1) > 5)
    test_times = test_data['event_times']
    test_types = test_data['event_types']
    test_morts = test_data['mortality']
    print(min(np.sum(test_data['event_types'], axis = 1)))

    val_times = val_data['event_times']
    val_types = val_data['event_types']
    val_morts = val_data['mortality']
    print(min(np.sum(val_data['event_types'], axis = 1)))

    train0id = (train_morts == 0)
    train1id = (train_morts == 1)

    train0_dataset = MixedDataset(train_vitals[train0id], train_vital_masks[train0id],
                        train_times[train0id], train_types[train0id], train_morts[train0id], train_statics[train0id], True)
    train1_dataset = MixedDataset(train_vitals[train1id], train_vital_masks[train1id],
                        train_times[train1id], train_types[train1id], train_morts[train1id], train_statics[train1id], True)
    val_dataset = MixedDataset(val_vitals, val_data['vital_masks'],
                        val_times, val_types, val_morts, val_data['statics'], True)
    test_dataset = MixedDataset(test_vitals, test_data['vital_masks'],
                        test_times, test_types, test_morts, test_data['statics'], True)
    
    ts_shape = train_vitals.shape
    static_shape = train_statics.shape

    return ts_shape, static_shape, [DataLoader(i, batch_size = batch_size, shuffle = True) for i in [train0_dataset, train1_dataset, val_dataset, test_dataset]]

def prepare_train_loader(dataname, batch_size):
    if dataname == 'mimic':
        train_data = np.load('./data/physionet/mimiciii_train.npz')
        vitals = np.concatenate((train_data['vitals'][:, :5, :], train_data['vitals'][:, 6:, :]), axis = 1)
    elif dataname == 'physionet':
        train_data = np.load('./data/physionet/physionet12_train.npz')
        vitals = train_data['vitals']
    else:
        train_data = np.load('./data/physionet/aumc_train.npz')
        vitals = train_data['vitals']

    vitals = np.clip(vitals, -2, 2)
    nums = np.sum(train_data['event_types'] > 0, axis = 1)
    
    event_times = train_data['event_times']
    event_times -= np.min(event_times, axis = 1).reshape((-1, 1))
    dataset = MixedDataset(vitals[nums > 24], train_data['vital_masks'][nums > 24], event_times[nums > 24], train_data['event_types'][nums > 24], train_data['mortality'][nums > 24])
    return DataLoader(dataset, batch_size = batch_size, shuffle=True)

def prepare_mimic_los(batch_size, regression = True):
    max_len = 150
    max_event_len = 400
    train_data = np.load('./data/physionet/mimiccut/mimiciii_train.npz')
    val_data = np.load('./data/physionet/mimiccut/mimiciii_val.npz')
    test_data = np.load('./data/physionet/mimiccut/mimiciii_test.npz')
    
    train_vitals = np.concatenate((train_data['vitals'][:, :5, :max_len], train_data['vitals'][:, 6:, :max_len]), axis = 1)
    val_vitals = np.concatenate((val_data['vitals'][:, :5, :max_len], val_data['vitals'][:, 6:, :max_len]), axis = 1)
    test_vitals = np.concatenate((test_data['vitals'][:, :5, :max_len], test_data['vitals'][:, 6:, :max_len]), axis = 1)

    train_times, train_types = decompose_events(train_data)
    train_times = train_times[:, :max_event_len]
    train_types = train_types[:, :max_event_len]

    train_idx = (np.sum(train_types>0, axis = 1) > 5)
    train_times = train_times[train_idx]
    train_times -= np.min(train_times, axis = 1).reshape((-1, 1))
    train_types = train_types[train_idx]
    train_morts = np.load('./data/physionet/mimiciii_train.npz')['los'][train_idx] - 2
    train_vitals = train_vitals[train_idx]
    train_vital_masks = train_data['vital_masks'][train_idx]
    train_statics = train_data['statics'][train_idx]
    # print(min(np.sum(train_types, axis = 1)))

    times, types = decompose_events(test_data)
    test_times = times[:, :max_event_len]
    test_types = types[:, :max_event_len]
    test_morts = np.load('./data/physionet/mimiciii_test.npz')['los'] - 2
    # print(min(np.sum(test_data['event_types'], axis = 1)))

    times, types = decompose_events(val_data)
    val_times = times[:, :max_event_len]
    val_types = types[:, :max_event_len]
    val_morts = np.load('./data/physionet/mimiciii_val.npz')['los'] - 2

    if not regression:
        train_morts = np.digitize(train_morts, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])
        train_morts = np.digitize(train_morts, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])
        train_morts = np.digitize(train_morts, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])

    train_dataset = MixedDataset(train_vitals, train_vital_masks,
                        train_times, train_types, train_morts, train_statics, True)
    val_dataset = MixedDataset(val_vitals, val_data['vital_masks'],
                        val_times, val_types, val_morts, val_data['statics'], True)
    test_dataset = MixedDataset(test_vitals, test_data['vital_masks'],
                        test_times, test_types, test_morts, test_data['statics'], True)
    
    ts_shape = train_vitals.shape
    static_shape = train_statics.shape

    return ts_shape, static_shape, [DataLoader(i, batch_size = batch_size, shuffle = True) for i in [train_dataset, val_dataset, test_dataset]]

def prepare_physionet_los(batch_size, regression = False):

    train_data = np.load('./data/physionet/physionet12_train.npz')
    val_data = np.load('./data/physionet/physionet12_val.npz')
    test_data = np.load('./data/physionet/physionet12_test.npz')
    train_vitals = train_data['vitals']
    val_vitals = val_data['vitals']
    test_vitals = test_data['vitals']

    train_vitals = np.clip(train_vitals, -2, 2)
    val_vitals = np.clip(val_vitals, -2, 2)
    test_vitals = np.clip(test_vitals, -2, 2)

    oc1 = pd.read_csv('./data/physionet/Outcomes-a.txt', usecols=['RecordID', 'Length_of_stay'])
    oc2 = pd.read_csv('./data/physionet/Outcomes-b.txt', usecols=['RecordID', 'Length_of_stay'])
    oc3 = pd.read_csv('./data/physionet/Outcomes-c.txt', usecols=['RecordID', 'Length_of_stay'])
    oc = pd.concat((oc1, oc2, oc3))
    # print(oc.keys())

    train_ids = pd.DataFrame({'RecordID': train_data['ids']})
    oc_train = pd.merge(train_ids, oc, how = 'left', on = 'RecordID')['Length_of_stay'].to_numpy() - 2
    test_ids = pd.DataFrame({'RecordID': test_data['ids']})
    oc_test = pd.merge(test_ids, oc, how = 'left', on = 'RecordID')['Length_of_stay'].to_numpy() - 2
    val_ids = pd.DataFrame({'RecordID': val_data['ids']})
    oc_val = pd.merge(val_ids, oc, how = 'left', on = 'RecordID')['Length_of_stay'].to_numpy() - 2

    if not regression:
        oc_train= np.digitize(oc_train-0.1, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])
        oc_test= np.digitize(oc_test-0.1, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])
        oc_val= np.digitize(oc_val-0.1, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])

    train_idx = (np.sum(train_data['event_types']>0, axis = 1) > 5)

    train_dataset = MixedDataset(train_vitals[train_idx], train_data['vital_masks'][train_idx],
            train_data['event_times'][train_idx], train_data['event_types'][train_idx], oc_train[train_idx], train_data['statics'][train_idx], True)
    val_dataset = MixedDataset(val_vitals, val_data['vital_masks'],
            val_data['event_times'], val_data['event_types'], oc_val, val_data['statics'], True)
    test_dataset = MixedDataset(test_vitals, test_data['vital_masks'],
            test_data['event_times'], test_data['event_types'], oc_test, test_data['statics'], True)
    
    ts_shape = train_vitals.shape
    static_shape = train_data['statics'].shape

    return ts_shape, static_shape, [DataLoader(i, batch_size = batch_size, shuffle = True) for i in [train_dataset, val_dataset, test_dataset]]

def prepare_aumc_los(batch_size, regression = False):
    train_data = np.load('./data/physionet/aumc_train.npz')
    val_data = np.load('./data/physionet/aumc_val.npz')
    test_data = np.load('./data/physionet/aumc_test.npz')
    train_vitals = train_data['vitals']
    val_vitals = val_data['vitals']
    test_vitals = test_data['vitals']

    train_vitals = np.clip(train_vitals, -2, 2)
    val_vitals = np.clip(val_vitals, -2, 2)
    test_vitals = np.clip(test_vitals, -2, 2)

    oc_train = train_data['los'] - 2
    oc_test = test_data['los'] - 2
    oc_val = val_data['los'] - 2

    if not regression:
        oc_train= np.digitize(oc_train, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])
        oc_test= np.digitize(oc_test, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])
        oc_val= np.digitize(oc_val, bins = [1, 2, 3, 4, 5, 6, 7, 8, 14])

        print(np.bincount(oc_train))
        print(np.bincount(oc_test))
        print(np.bincount(oc_val))
    train_idx = (np.sum(train_data['event_types']>0, axis = 1) > 5)

    train_dataset = MixedDataset(train_vitals[train_idx], train_data['vital_masks'][train_idx],
            train_data['event_times'][train_idx], train_data['event_types'][train_idx], oc_train[train_idx], train_data['statics'][train_idx], True)
    val_dataset = MixedDataset(val_vitals, val_data['vital_masks'],
            val_data['event_times'], val_data['event_types'], oc_val, val_data['statics'], True)
    test_dataset = MixedDataset(test_vitals, test_data['vital_masks'],
            test_data['event_times'], test_data['event_types'], oc_test, test_data['statics'], True)
    
    ts_shape = train_vitals.shape
    static_shape = train_data['statics'].shape

    return ts_shape, static_shape, [DataLoader(i, batch_size = batch_size, shuffle = True) for i in [train_dataset, val_dataset, test_dataset]]

def prepare_pretrain_loader(dataname, batch_size):

    if dataname == 'mimic':
        train_data = np.load('./data/physionet/physionet12_train.npz')
        val_data = np.load('./data/physionet/physionet12_val.npz')
        test_data = np.load('./data/physionet/physionet12_test.npz')

        vitals = np.concatenate([train_data['vitals'], val_data['vitals'], test_data['vitals']], axis = 0)
        vital_masks = np.concatenate([train_data['vital_masks'], val_data['vital_masks'], test_data['vital_masks']], axis = 0)
        event_types = np.concatenate([train_data['event_types'], val_data['event_types'], test_data['event_types']], axis = 0)
        event_times = np.concatenate([train_data['event_times'], val_data['event_times'], test_data['event_times']], axis = 0)
        mortality = np.concatenate([train_data['mortality'], val_data['mortality'], test_data['mortality']], axis = 0)

        train_data = np.load('./data/physionet/aumc_train.npz')
        val_data = np.load('./data/physionet/aumc_val.npz')
        test_data = np.load('./data/physionet/aumc_test.npz')

        vitals = np.concatenate([vitals, train_data['vitals'], val_data['vitals'], test_data['vitals']], axis = 0)
        vital_masks = np.concatenate([vital_masks, train_data['vital_masks'], val_data['vital_masks'], test_data['vital_masks']], axis = 0)
        event_types = np.concatenate([event_types, train_data['event_types'], val_data['event_types'], test_data['event_types']], axis = 0)
        event_times = np.concatenate([event_times, train_data['event_times'], val_data['event_times'], test_data['event_times']], axis = 0)
        mortality = np.concatenate([mortality, train_data['mortality'], val_data['mortality'], test_data['mortality']], axis = 0)
    
    elif dataname == 'physionet':
        train_data = np.load('./data/physionet/mimiciii_train.npz')
        val_data = np.load('./data/physionet/mimiciii_val.npz')
        test_data = np.load('./data/physionet/mimiciii_test.npz')

        vitals = np.concatenate([train_data['vitals'], val_data['vitals'], test_data['vitals']], axis = 0)
        vitals = np.concatenate((vitals[:, :5, :], vitals[:, 6:, :]), axis = 1)
        vital_masks = np.concatenate([train_data['vital_masks'], val_data['vital_masks'], test_data['vital_masks']], axis = 0)
        event_types = np.concatenate([train_data['event_types'], val_data['event_types'], test_data['event_types']], axis = 0)
        event_times = np.concatenate([train_data['event_times'], val_data['event_times'], test_data['event_times']], axis = 0)
        mortality = np.concatenate([train_data['mortality'], val_data['mortality'], test_data['mortality']], axis = 0)

        train_data = np.load('./data/physionet/aumc_train.npz')
        val_data = np.load('./data/physionet/aumc_val.npz')
        test_data = np.load('./data/physionet/aumc_test.npz')

        vitals = np.concatenate([vitals, train_data['vitals'], val_data['vitals'], test_data['vitals']], axis = 0)
        vital_masks = np.concatenate([vital_masks, train_data['vital_masks'], val_data['vital_masks'], test_data['vital_masks']], axis = 0)
        event_types = np.concatenate([event_types, train_data['event_types'], val_data['event_types'], test_data['event_types']], axis = 0)
        event_times = np.concatenate([event_times, train_data['event_times'], val_data['event_times'], test_data['event_times']], axis = 0)
        mortality = np.concatenate([mortality, train_data['mortality'], val_data['mortality'], test_data['mortality']], axis = 0)
    else:
        train_data = np.load('./data/physionet/mimiciii_train.npz')
        val_data = np.load('./data/physionet/mimiciii_val.npz')
        test_data = np.load('./data/physionet/mimiciii_test.npz')

        vitals = np.concatenate([train_data['vitals'], val_data['vitals'], test_data['vitals']], axis = 0)
        vitals = np.concatenate((vitals[:, :5, :], vitals[:, 6:, :]), axis = 1)
        vital_masks = np.concatenate([train_data['vital_masks'], val_data['vital_masks'], test_data['vital_masks']], axis = 0)
        event_types = np.concatenate([train_data['event_types'], val_data['event_types'], test_data['event_types']], axis = 0)
        event_times = np.concatenate([train_data['event_times'], val_data['event_times'], test_data['event_times']], axis = 0)
        mortality = np.concatenate([train_data['mortality'], val_data['mortality'], test_data['mortality']], axis = 0)

        train_data = np.load('./data/physionet/physionet12_train.npz')
        val_data = np.load('./data/physionet/physionet12_val.npz')
        test_data = np.load('./data/physionet/physionet12_test.npz')

        vitals = np.concatenate([vitals, train_data['vitals'], val_data['vitals'], test_data['vitals']], axis = 0)
        vital_masks = np.concatenate([vital_masks, train_data['vital_masks'], val_data['vital_masks'], test_data['vital_masks']], axis = 0)
        event_types = np.concatenate([event_types, train_data['event_types'], val_data['event_types'], test_data['event_types']], axis = 0)
        event_times = np.concatenate([event_times, train_data['event_times'], val_data['event_times'], test_data['event_times']], axis = 0)
        mortality = np.concatenate([mortality, train_data['mortality'], val_data['mortality'], test_data['mortality']], axis = 0)

    event_times -= np.min(event_times, axis = 1).reshape((-1, 1))
    print(np.nonzero(np.min(event_times, axis = 1)))
    print(np.min(vitals), np.max(vitals))
    vitals = np.clip(vitals, -2, 2)

    nums = np.sum(event_types > 0, axis = 1)

    return DataLoader(MixedDataset(vitals[nums > 50], vital_masks[nums > 50], event_times[nums > 50], event_types[nums > 50], mortality[nums > 50]), batch_size = batch_size)
