from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
from ts_transformer.ts_transformer import TSTransformerEncoder, DownstreamClassifier, MaskedLanguageModel
from THP.Models import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
from pipeline.utils import *
from pipeline.supervised import *
from pipeline.unsupervised import *
import sys

norm = 'LayerNorm'
lr = 0.001
gamma = 0
l2_coef = 0
record_freq = 2
decay_freq = 20
downstream_freq = 25
batch_size = 256
d_model = 128
d_inner = 128
n_heads = 2
n_event_heads = 2
num_layers = 1
num_event_layers = 1
dropout = 0.1
dim_feedforward = 128
max_ts_len = 150
max_event_len = 400
aggr = 'max'

num_events = 29

emb, task, dataname, downs_dataname = sys.argv[1:]

if task == 'Supervised':
        now = datetime.now()
        tb = SummaryWriter(log_dir='./log/{}_{}_{}_{}_batch{}_lr{}_gm{}_freq{}_l2{}_{}_dm{}_df{}_di{}_drop{}_nheads{}_nlayers{}_AGGR{}'.format(dataname, emb, task, now.strftime('%m.%d'),\
        batch_size, lr, gamma, record_freq, l2_coef, norm, d_model, dim_feedforward, d_inner, dropout, n_heads, num_layers, aggr), filename_suffix='basic_setting', )

        ts_shape, static_shape, loaders = prepare_balanced_loaders(dataname, batch_size, max_ts_len, max_event_len)
        print('Using dataset of size (train, val, test): ', [len(i) for i in loaders], '*', batch_size)
        print(ts_shape, static_shape)
        if emb == 'TS':
                downstream_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)
                model = DownstreamClassifier(downstream_encoder, d_model + static_shape[1], 2, aggr).cuda()
                torch.nn.init.xavier_normal_(model.linear.weight)
                # for param in model.encoder.parameters():
                #         param.requires_grad = False
                supervised_run(model, loaders, tb, lr, l2_coef, record_freq, False, total_epoch = 200)
        else: # Mixed
                ts_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)

                event_encoder = Encoder(num_events, d_model, d_inner, num_event_layers, n_event_heads, dim_feedforward, dim_feedforward, dropout)
                model = MixedClassifier(event_encoder, ts_encoder, d_model, d_model, static_shape[1], 2, aggr).cuda()
                mixed_finetune_balanced(model, loaders, tb, lr, torch.nn.CrossEntropyLoss(), record_freq, 200, l2_coef)
elif task == 'Pheno':
        now = datetime.now()
        tb = SummaryWriter(log_dir='./log/Pheno_{}_batch{}_lr{}_gm{}_freq{}_l2{}_{}_dm{}_df{}_di{}_drop{}_nheads{}_nlayers{}_AGGR{}'.format(now.strftime('%m.%d'),\
        batch_size, lr, gamma, record_freq, l2_coef, norm, d_model, dim_feedforward, d_inner, dropout, n_heads, num_layers, aggr), filename_suffix='basic_setting', )

        ratio = 50
        ts_shape, static_shape, loaders = prepare_pheno(batch_size, max_event_len, ratio)
        pretrain_loader = loaders[0]

        if ratio != 100:
                loaders = (loaders[1], loaders[2], loaders[3])
        print('Using dataset of size (train, val, test): ', [len(i) for i in loaders], '*', batch_size)
        print('Using pretrain dataset of size : ', len(pretrain_loader), '*', batch_size)
        ts_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)

        event_encoder = Encoder(num_events, d_model, d_inner, num_event_layers, n_event_heads, dim_feedforward, dim_feedforward, dropout)
        downs_model = MixedClassifier(event_encoder, ts_encoder, d_model, d_model, static_shape[1], 25, aggr).cuda()
        
        generator_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= int(d_model), n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward= dim_feedforward, norm = norm)
        generator = MaskedLanguageModel(generator_encoder, int(d_model), ts_shape[1], aggr).cuda()

        discriminator_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)
        discriminator = DownstreamClassifier(discriminator_encoder, d_model, num_classes=2*ts_shape[1], aggr = 'none').cuda()

        hawkes = Transformer(num_events, d_model, d_model, d_inner, num_layers, n_heads, dim_feedforward, dim_feedforward, dropout).cuda()

        model = ELECTRA_finalized(generator, discriminator, hawkes).cuda()

        # event_encoder = Encoder(num_events, d_model, d_inner, num_event_layers, n_event_heads, dim_feedforward, dim_feedforward, dropout)
        # model = MixedClassifier(event_encoder, ts_encoder, d_model, d_model, static_shape[1], 25, aggr).cuda()
        # model = MultiLabelClassifier(event_encoder, ts_encoder, d_model, d_model, static_shape[1], 25, aggr).cuda()
        # L1_L2_L3_pretrain(model, pretrain_loader, tb, lr, record_freq, downstream_freq, 30, False)
        # downs_model.ts_encoder.load_state_dict(model.generator.encoder.state_dict())
        # downs_model.event_encoder.load_state_dict(model.hawkes.encoder.state_dict())

        pheno_finetune(downs_model, loaders, tb, lr, record_freq, 200, 0)

        # evaluate_pheno(model, loaders[2], 'eval', None, 0)

elif task == 'LOS':
        now = datetime.now()
        tb = SummaryWriter(log_dir='./log/{}_{}_{}_{}_batch{}_lr{}_gm{}_freq{}_l2{}_{}_dm{}_df{}_di{}_drop{}_nheads{}_nlayers{}_AGGR{}'.format(dataname, emb, task, now.strftime('%m.%d'),\
        batch_size, lr, gamma, record_freq, l2_coef, norm, d_model, dim_feedforward, d_inner, dropout, n_heads, num_layers, aggr), filename_suffix='basic_setting', )

        regression = True
        if dataname == 'mimic':
                ts_shape, static_shape, loaders = prepare_mimic_los(batch_size, regression)
        elif dataname == 'physionet':
                ts_shape, static_shape, loaders = prepare_physionet_los(batch_size, regression)
        elif dataname == 'aumc':
                ts_shape, static_shape, loaders = prepare_aumc_los(batch_size, regression)
        # pretrain_loader = prepare_train_loader(dataname, batch_size)
        pretrain_loader = prepare_pretrain_loader(dataname, batch_size)
        print(ts_shape, static_shape)
        print('Using dataset of size (train, val, test): ', [len(i) for i in loaders], '*', batch_size)

        ts_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)

        event_encoder = Encoder(num_events, d_model, d_inner, num_event_layers, n_event_heads, dim_feedforward, dim_feedforward, dropout)

        if regression:
                num_classes = 1
        else:
                num_classes = 10
        downs_model = MixedClassifier(event_encoder, ts_encoder, d_model, d_model, static_shape[1], num_classes, aggr).cuda()

        pretrain_epoch = 30
        downstream_freq = pretrain_epoch - 1
        if emb == 'Supervised':
                LOS_finetune(downs_model, loaders, tb, 0.001, torch.nn.CrossEntropyLoss(), record_freq, 200, l2_coef, regression = regression)
        elif emb == 'TS_ELECTRA':
                generator_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= int(d_model), n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward= dim_feedforward, norm = norm)
                generator = MaskedLanguageModel(generator_encoder, int(d_model), ts_shape[1], aggr).cuda()

                discriminator_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)
                discriminator = DownstreamClassifier(discriminator_encoder, d_model, num_classes=2*ts_shape[1], aggr = 'none').cuda()

                hawkes = Transformer(num_events, d_model, d_model, d_inner, num_layers, n_heads, dim_feedforward, dim_feedforward, dropout).cuda()

                model = ELECTRA_finalized(generator, discriminator, hawkes).cuda()

                L1_L2_L3_pretrain(model, pretrain_loader, tb, lr, record_freq, downstream_freq, pretrain_epoch, False)
                downs_model.ts_encoder.load_state_dict(model.generator.encoder.state_dict())
                downs_model.event_encoder.load_state_dict(model.hawkes.encoder.state_dict())
                LOS_finetune(downs_model, loaders, tb, 0.001, torch.nn.CrossEntropyLoss(), record_freq, 200, l2_coef, regression = regression)
else: # Transfer
        now = datetime.now()
        tb = SummaryWriter(log_dir='./log/{}_{}_{}_{}_{}_batch{}_lr{}_gm{}_freq{}_l2{}_{}_dm{}_df{}_drop{}_nheads{}_nlayers{}_AGGR{}'.format(dataname, emb, task, downs_dataname, now.strftime('%m.%d'),\
        batch_size, lr, gamma, record_freq, l2_coef, norm, d_model, dim_feedforward, dropout, n_heads, num_layers, aggr), filename_suffix='basic_setting', )

        ts_shape, static_shape, downs_loaders = prepare_balanced_loaders(downs_dataname, batch_size, max_ts_len, max_event_len)

        # pretrain_loader = prepare_train_loader(dataname, batch_size)
        pretrain_loader = prepare_pretrain_loader(dataname, batch_size)

        print('Using pretrain dataset of size: ', len(pretrain_loader), '*', batch_size)
        print('Using downstream dataset of size (train, val, test): ', [len(i) for i in downs_loaders], '*', batch_size)
        print(ts_shape, static_shape)

        pretrain_epoch = 500

        if emb == 'Missing':
                generator_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= int(d_model), n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward= dim_feedforward, norm = norm)
                generator = MaskedLanguageModel(generator_encoder, int(d_model), ts_shape[1], aggr).cuda()

                downs_ts_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)
                downs_event_encoder = Encoder(num_events, d_model, d_inner, num_event_layers, n_event_heads, dim_feedforward, dim_feedforward, dropout)
                downs_model = MixedClassifier(downs_event_encoder, downs_ts_encoder, d_model, d_model, static_shape[1], 2, aggr).cuda()
                L1_pretrain(generator, pretrain_loader, tb, lr, record_freq, downstream_freq, pretrain_epoch, True, downs_model, downs_loaders)

        elif emb == 'Event':
                model = Transformer(num_events, d_model, d_model, d_inner, num_layers, n_heads, dim_feedforward, dim_feedforward, dropout).cuda()
                
                downs_ts_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)
                downs_event_encoder = Encoder(num_events, d_model, d_inner, num_event_layers, n_event_heads, dim_feedforward, dim_feedforward, dropout)
                downs_model = MixedClassifier(downs_event_encoder, downs_ts_encoder, d_model, d_model, static_shape[1], 2, aggr).cuda()
                L3_pretrain(model, pretrain_loader, tb, lr, record_freq, downstream_freq, pretrain_epoch, True, downs_model, downs_loaders)

        elif emb == 'TS_ELECTRA':
                generator_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= int(d_model), n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward= dim_feedforward, norm = norm)
                generator = MaskedLanguageModel(generator_encoder, int(d_model), ts_shape[1], aggr).cuda()

                discriminator_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)
                discriminator = DownstreamClassifier(discriminator_encoder, d_model, num_classes=2*ts_shape[1], aggr = 'none').cuda()

                hawkes = Transformer(num_events, d_model, d_model, d_inner, num_layers, n_heads, dim_feedforward, dim_feedforward, dropout).cuda()

                model = ELECTRA_finalized(generator, discriminator, hawkes).cuda()

                downs_ts_encoder = TSTransformerEncoder(ts_shape[1], ts_shape[2], \
                        d_model= d_model, n_heads=n_heads, num_layers=num_layers, dropout= dropout, dim_feedforward=dim_feedforward, norm = norm)
                downs_event_encoder = Encoder(num_events, d_model, d_inner, num_event_layers, n_event_heads, dim_feedforward, dim_feedforward, dropout)
                downs_model = MixedClassifier(downs_event_encoder, downs_ts_encoder, d_model, d_model, static_shape[1], 2, aggr).cuda()

                L1_L2_L3_pretrain(model, pretrain_loader, tb, lr, record_freq, downstream_freq, pretrain_epoch, False)
                downs_model.ts_encoder.load_state_dict(model.generator.encoder.state_dict())
                downs_model.event_encoder.load_state_dict(model.hawkes.encoder.state_dict())

                # for param in downs_model.event_encoder.parameters():
                #         param.requires_grad = False
                # for param in downs_model.ts_encoder.parameters():
                #         param.requires_grad = False
                
                train_scores, test_scores, step = mixed_finetune_balanced(downs_model, downs_loaders, None, 0.0005, 
                                        torch.nn.CrossEntropyLoss(), 2, 100)

        else: 
                raise ValueError('Error task')


        