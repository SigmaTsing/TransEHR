import torch 
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pipeline.utils import InfIter, MaskedMSELoss
from pipeline.supervised import *
import pipeline.thp_utils as Utils
from sklearn.metrics import accuracy_score

def L1_pretrain(model: torch.nn.Module, loader, writer, learning_rate, record_freq, downstream_freq = 10, total_epoch = 100, finetune = False, downstream_model:torch.nn.Module = None, downstream_loaders = None):
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

    losses = []
    masking_ratio = 0.15
    for epoch in (pbar := tqdm(range(total_epoch))):
        for batch in loader:
            x, padding_mask, _, _, _ = map(lambda x: x.cuda(), batch)
            x = x.float().transpose(2, 1)
            ts_masks = np.random.choice(np.array([True, False]), size=x.shape, replace=True,
                                    p=(masking_ratio, 1- masking_ratio))
            ts_masks = torch.from_numpy(ts_masks).cuda()
            
            gen_out = model(x*(~ts_masks), padding_mask)
            target = torch.masked_select(x, ts_masks*padding_mask.unsqueeze(-1))

            loss = F.mse_loss(torch.masked_select(gen_out, ts_masks*padding_mask.unsqueeze(-1))[torch.nonzero(target)], target[torch.nonzero(target)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ update parameters """
            losses.append(loss.item())
        if epoch % record_freq == record_freq - 1:
            writer.add_scalars('Loss', {'train': np.mean(losses)}, global_step = epoch) 
            pbar.set_description('Loss {}'.format(np.mean(losses)))
            losses.clear()
            
        if epoch % downstream_freq == downstream_freq -1 and finetune:
            downstream_model.ts_encoder.load_state_dict(model.encoder.state_dict())
            
            train_scores, test_scores, step = mixed_finetune_balanced(downstream_model, downstream_loaders, None, 0.0005, 
                                    torch.nn.CrossEntropyLoss(), 2, 100)

            acc, roc, prc = train_scores
            writer.add_scalars('Accuracy', {'discriminator_train': acc}, global_step = epoch) 
            writer.add_scalars('AUROC', {'discriminator_train': roc}, global_step = epoch)     
            writer.add_scalars('AUPRC', {'discriminator_train': prc}, global_step = epoch)
            acc, roc, prc = test_scores
            writer.add_scalars('Accuracy', {'discriminator_test': acc}, global_step = epoch) 
            writer.add_scalars('AUROC', {'discriminator_test': roc}, global_step = epoch)     
            writer.add_scalars('AUPRC', {'discriminator_test': prc}, global_step = epoch)

def L1_L2_L3_pretrain(model, loader, writer, learning_rate, record_freq, downstream_freq = 10, total_epoch = 100, finetune = False, downstream_model:torch.nn.Module = None, downstream_loaders = None):
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    
    losses = []
    gen_losses = []
    disc_losses = []
    nll_losses = []

    disc_weight = 0.5
    nll_weight = 1e-6

    masking_ratio = 0.25
    for epoch in (pbar := tqdm(range(total_epoch))):
        for batch in loader:
            x, padding_mask, event_times, event_types, _ , _= map(lambda x: x.cuda(), batch)
            x = x.float().transpose(2, 1)

            ts_masks = np.random.choice(np.array([True, False]), size=x.shape, replace=True,
                                    p=(masking_ratio, 1- masking_ratio))
            ts_masks = torch.from_numpy(ts_masks).cuda()

            leng = torch.sum(padding_mask, dim = -1).long() - 1

            forecast_mask = torch.zeros_like(padding_mask)
            forecast_mask = forecast_mask.scatter_(1, leng.view(-1, 1), 1, reduce = 'add').cuda()
            total_mask = ts_masks.bool() | forecast_mask.bool().unsqueeze(-1)
            
            event_enc, gen_loss, disc_loss = model(x, padding_mask, total_mask, event_times, event_types.long())
            event_ll, non_event_ll = Utils.log_likelihood(model.hawkes, event_enc, event_times, event_types.long())
            nll_loss = -torch.sum(event_ll - non_event_ll)

            loss = gen_loss + disc_weight* disc_loss + nll_loss * nll_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ update parameters """
            losses.append(loss.item())

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item()* disc_weight)
            nll_losses.append(nll_loss.item()* nll_weight)

        # if epoch % record_freq == record_freq - 1:
        writer.add_scalars('Loss', {'GEN': np.mean(gen_losses)}, global_step = epoch) 
        writer.add_scalars('Loss', {'DISC': np.mean(disc_losses)}, global_step = epoch) 
        writer.add_scalars('Loss', {'NLL': np.mean(nll_losses)}, global_step = epoch) 
        pbar.set_description('Loss: GEN {}; DISC {} NLL {}'.format(np.mean(gen_losses), np.mean(disc_losses), np.mean(nll_losses)))
        losses.clear()
        gen_losses.clear()
        disc_losses.clear()
        nll_losses.clear()
            
        if epoch % downstream_freq == downstream_freq -1 and finetune:
            downstream_model.ts_encoder.load_state_dict(model.discriminator.encoder.state_dict())
            downstream_model.event_encoder.load_state_dict(model.hawkes.encoder.state_dict())
            
            train_scores, test_scores, step = mixed_finetune_balanced(downstream_model, downstream_loaders, None, 0.0005, 
                                    torch.nn.CrossEntropyLoss(), 2, 100)

            acc, roc, prc = train_scores
            writer.add_scalars('Accuracy', {'downstream_train': acc}, global_step = epoch) 
            writer.add_scalars('AUROC', {'downstream_train': roc}, global_step = epoch)     
            writer.add_scalars('AUPRC', {'downstream_train': prc}, global_step = epoch)
            acc, roc, prc = test_scores
            writer.add_scalars('Accuracy', {'downstream_test': acc}, global_step = epoch) 
            writer.add_scalars('AUROC', {'downstream_test': roc}, global_step = epoch)     
            writer.add_scalars('AUPRC', {'downstream_test': prc}, global_step = epoch)

def L3_pretrain(model, loader, writer, learning_rate, record_freq, downstream_freq = 10, total_epoch = 100, finetune = False, downstream_model:torch.nn.Module = None, downstream_loaders = None):
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    
    losses = []
    gen_losses = []
    disc_losses = []
    nll_losses = []

    disc_weight = 0.5
    nll_weight = 1e-6

    for epoch in (pbar := tqdm(range(total_epoch))):
        for batch in loader:
            _, _, event_times, event_types, _ = map(lambda x: x.cuda(), batch)

            event_types = event_types.long()
            enc_out, prediction = model(event_types, event_times)

            """ backward """
            # negative log-likelihood
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_times, event_types)
            event_loss = -torch.sum(event_ll - non_event_ll)

            # type prediction
            pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_types, torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none'))

            loss = event_loss + pred_loss
            # loss = pred_loss + event_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ update parameters """
            losses.append(loss.item())

            gen_losses.append(event_loss.item())
            disc_losses.append(pred_loss.item())
            # nll_losses.append(nll_loss.item()* nll_weight)

        # if epoch % record_freq == record_freq - 1:
        # writer.add_scalars('Loss', {'GEN': np.mean(gen_losses)}, global_step = epoch) 
        # writer.add_scalars('Loss', {'DISC': np.mean(disc_losses)}, global_step = epoch) 
        # writer.add_scalars('Loss', {'NLL': np.mean(nll_losses)}, global_step = epoch) 
        pbar.set_description('Loss: EVENT {} TYPE {}'.format(np.mean(gen_losses), np.mean(disc_losses)))
        losses.clear()
        gen_losses.clear()
        disc_losses.clear()
        nll_losses.clear()
            
        if epoch % downstream_freq == downstream_freq -1 and finetune:
            # downstream_model.ts_encoder.load_state_dict(model.generator.encoder.state_dict())
            downstream_model.event_encoder.load_state_dict(model.encoder.state_dict())
            
            train_scores, test_scores, step = mixed_finetune_balanced(downstream_model, downstream_loaders, None, 0.0005, 
                                    torch.nn.CrossEntropyLoss(), 2, 100)

            acc, roc, prc = train_scores
            writer.add_scalars('Accuracy', {'downstream_train': acc}, global_step = epoch) 
            writer.add_scalars('AUROC', {'downstream_train': roc}, global_step = epoch)     
            writer.add_scalars('AUPRC', {'downstream_train': prc}, global_step = epoch)
            acc, roc, prc = test_scores
            writer.add_scalars('Accuracy', {'downstream_test': acc}, global_step = epoch) 
            writer.add_scalars('AUROC', {'downstream_test': roc}, global_step = epoch)     
            writer.add_scalars('AUPRC', {'downstream_test': prc}, global_step = epoch)
