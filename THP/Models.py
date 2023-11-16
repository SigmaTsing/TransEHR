import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import THP.Constants as Constants
from THP.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths.cpu(), batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        # enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)

class Classifier(nn.Module):
    def __init__(
            self,
            num_types, d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, d_statics = 7, dropout=0.1, aggr = 'max'):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types
        self.downstream_predictor = torch.nn.Linear(d_model + d_statics, 2)
        self.aggr = aggr

    def forward(self, event_type, event_time, statics):
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        if self.aggr == 'max':
            enc_output, _ = torch.max(enc_output, dim = 1)
        elif self.aggr == 'mean':
            enc_output = torch.mean(enc_output, dim = 1)
        enc_output = torch.cat((enc_output, statics), dim = 1)
        prediction = self.downstream_predictor(enc_output)

        return enc_output, prediction

class THP_ELECTRA(nn.Module):
    def __init__(
            self, generator: Transformer, discriminator:Transformer):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, event_type, event_time):
        #TODO: add disc loss based on seq_to_seq output
        enc_out, prediction = self.generator(event_type, event_time)
        
        pred_type = prediction[0][:, :-1, :].detach()
        pred_type = torch.max(pred_type, dim=-1)[1] +1
        pred_time = prediction[1].squeeze_(-1)[:, :-1].detach()
        
        # disc_out, disc_pred = self.discriminator(pred_type, pred_time)
        disc_out, disc_pred = self.discriminator(pred_type, event_time[:, 1:])
        return enc_out, prediction, disc_pred

class EventMaskedModel(nn.Module):
    def __init__(
            self, encoder: Encoder, d_model, num_types):
        super().__init__()
        self.encoder = encoder
        self.type_predictor = nn.Linear(d_model, num_types)
    
    def forward(self, event_type, event_time):
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        type_prediction = self.type_predictor(enc_output)
        return type_prediction

class EventELECTRA(nn.Module):
    def __init__(
            self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def forward(self, event_type, event_time):
        type_pred = self.generator(event_type, event_time)
        disc_input = torch.max(type_pred, dim = -1)[1] + 1
        disc_pred = self.discriminator(disc_input, event_time)
        return type_pred, disc_input, disc_pred

class ModifiedELECTRA(nn.Module):
    def __init__(
            self, generator, discriminator, ts_model):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.ts_model = ts_model
    
    def forward(self, x, padding_mask, event_type, event_time):
        type_pred = self.generator(event_type, event_time)
        disc_input = torch.max(type_pred, dim = -1)[1] + 1
        disc_pred = self.discriminator(disc_input, event_time)
        ts_output = self.ts_model(x, padding_mask)
        return ts_output, type_pred, disc_input, disc_pred

class L1f_L2_L3(nn.Module):
    def __init__(self, generator, discriminator, hawkes):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.hawkes = hawkes

    def forward(self, x, padding_mask, forecast_mask, event_time, event_type):
        # non_pad_mask = get_non_pad_mask(event_type)
        # if torch.sum(non_pad_mask) == 0:
        #     raise ValueError('ERROR: zero mask')
        event_enc, _ = self.hawkes(event_type.long(), event_time)

        gen_out = self.generator(x, padding_mask*(~forecast_mask))
        gen_target = torch.masked_select(x, forecast_mask.unsqueeze(-1))
        gen_loss = F.mse_loss(torch.masked_select(gen_out, forecast_mask.unsqueeze(-1))[torch.nonzero(gen_target)], gen_target[torch.nonzero(gen_target)])

        x[forecast_mask] = gen_out[forecast_mask]
        disc_out = self.discriminator(x, padding_mask)
        disc_out = disc_out[padding_mask == 1]
        disc_label = forecast_mask.long()[padding_mask == 1]
        disc_loss = F.cross_entropy(disc_out, disc_label)

        return event_enc, gen_loss, disc_loss

class ELECTRA_finalized(nn.Module):
    def __init__(self, generator, discriminator, hawkes):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.hawkes = hawkes

    def forward(self, x, padding_mask, target_mask, event_time, event_type):
        # non_pad_mask = get_non_pad_mask(event_type)
        # if torch.sum(non_pad_mask) == 0:
        #     raise ValueError('ERROR: zero mask')
        event_enc, _ = self.hawkes(event_type.long(), event_time)

        gen_out = self.generator(x*(~target_mask), padding_mask)
        gen_target = torch.masked_select(x, target_mask*padding_mask.unsqueeze(-1))
        gen_loss = F.mse_loss(torch.masked_select(gen_out, target_mask*padding_mask.unsqueeze(-1))[torch.nonzero(gen_target)], gen_target[torch.nonzero(gen_target)])

        x[target_mask] = gen_out[target_mask]
        disc_out = self.discriminator(x, padding_mask)
        disc_out = disc_out.reshape((disc_out.shape[0], disc_out.shape[1], 2, -1))
        # print(disc_out.shape)
        disc_out = disc_out[padding_mask == 1]
        disc_label = target_mask.long()[padding_mask == 1]
        # print(disc_label.shape, disc_out.shape)
        disc_loss = F.cross_entropy(disc_out, disc_label)

        return event_enc, gen_loss, disc_loss

class ELECTRA_re(nn.Module):
    def __init__(self, generator, discriminator, hawkes):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.hawkes = hawkes

    def forward(self, x, padding_mask, target_mask, event_time, event_type):
        # non_pad_mask = get_non_pad_mask(event_type)
        # if torch.sum(non_pad_mask) == 0:
        #     raise ValueError('ERROR: zero mask')
        event_enc, _ = self.hawkes(event_type.long(), event_time)

        gen_out = self.generator(x, padding_mask*(~target_mask))
        gen_target = torch.masked_select(x, (target_mask*padding_mask).unsqueeze(-1))
        gen_loss = F.mse_loss(torch.masked_select(gen_out, (target_mask*padding_mask).unsqueeze(-1))[torch.nonzero(gen_target)], gen_target[torch.nonzero(gen_target)])

        x[target_mask] = gen_out[target_mask]
        disc_out = self.discriminator(x, padding_mask)
        # print(disc_out.shape)
        disc_out = disc_out[padding_mask == 1]
        disc_label = target_mask.long()[padding_mask == 1]
        # print(disc_label.shape, disc_out.shape)
        disc_loss = F.cross_entropy(disc_out, disc_label)

        return event_enc, gen_loss, disc_loss

class MixedClassifier(torch.nn.Module):
    def __init__(self, event_encoder, ts_encoder, d_event, d_ts, d_statics, num_classes, aggr = 'max'):
        super().__init__()
        self.event_encoder = event_encoder
        self.ts_encoder = ts_encoder
        self.linear = torch.nn.Linear(d_event+d_ts+d_statics, 32)
        self.linear1 = torch.nn.Linear(32, num_classes)
        self.aggr = aggr

    def forward(self, event_time, event_type, TS, padding_masks, statics = None):
        non_pad_mask = get_non_pad_mask(event_type)
        event_enc = self.event_encoder(event_type, event_time, non_pad_mask)
        if self.aggr == 'max':
            event_enc, _ = torch.max(event_enc, dim = 1)
        else:
            event_enc = torch.mean(event_enc, dim = 1)

        ts_enc = self.ts_encoder(TS, padding_masks)
        if self.aggr == 'max':
            ts_enc = ts_enc * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
            ts_enc, _ = torch.max(ts_enc, dim = 1)
        else:
            ts_enc = torch.mean(ts_enc, dim = 1)
        # print('ts', np.sum(np.isnan(ts_enc.cpu().detach().numpy())))
        # print('event', np.sum(np.isnan(event_enc.cpu().detach().numpy())))
        if statics is not None:
            # print('static', np.sum(np.isnan(statics.cpu().detach().numpy())))
            enc = torch.cat((event_enc, ts_enc, statics), dim = 1)
        else:
            enc = torch.cat((event_enc, ts_enc), dim = 1)
        # print('conc', np.sum(np.isnan(enc.cpu().detach().numpy())))
        enc = self.linear(enc)
        # print('enc', np.sum(np.isnan(enc.cpu().detach().numpy())))
        return self.linear1(F.gelu(enc))

class MultiLabelClassifier(torch.nn.Module):
    def __init__(self, event_encoder, ts_encoder, d_event, d_ts, d_statics, head_count, aggr = 'max'):
        super().__init__()
        self.event_encoder = event_encoder
        self.ts_encoder = ts_encoder
        self.heads = []
        for i in range(head_count):
            self.heads.append(torch.nn.Linear(d_event+d_ts+d_statics, 1).cuda())
        self.aggr = aggr

    def forward(self, event_time, event_type, TS, padding_masks, statics):
        non_pad_mask = get_non_pad_mask(event_type)
        event_enc = self.event_encoder(event_type, event_time, non_pad_mask)
        if self.aggr == 'max':
            event_enc, _ = torch.max(event_enc, dim = 1)
        else:
            event_enc = torch.mean(event_enc, dim = 1)

        ts_enc = self.ts_encoder(TS, padding_masks)
        if self.aggr == 'max':
            ts_enc = ts_enc * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
            ts_enc, _ = torch.max(ts_enc, dim = 1)
        else:
            ts_enc = torch.mean(ts_enc, dim = 1)

        enc = torch.cat((event_enc, ts_enc, statics), dim = 1)
        return torch.cat([i(enc) for i in self.heads], dim = 1)