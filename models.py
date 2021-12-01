from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel


class MusicTransformer(BaseModel):
    """
        Transformer architecture for motifs inspired by
        C.-Z. A. Huang, A. Vaswani, J. Uszkoreit, N. Shazeer,
        I. Simon, C. Hawthorne, A. M. Dai, M. D. Hoffman,
        M. Dinculescu and D. Eck
        "Music Transformer"
        https://arxiv.org/abs/1809.04281
    """
    def __init__(
        self,
        att_filters=None,
        device=torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        channels=128,
        verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if att_filters is None:
            self.att_filters = [64, 32, 16]
        else:
            self.att_filters = att_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        encoder_channels = [channels] + self.att_filters

        # <Parameter setup>
        self.encoder = nn.ModuleList([
            nn.Sequential(
                MultiheadedAttention(f_in, f_out),
                nn.BatchNorm1d(f_out)
            )
            for f_in, f_out in zip(
                encoder_channels[:-1], encoder_channels[1:]
            )
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(
                MultiheadedAttention(f_in, f_out),
                nn.BatchNorm1d(f_out)
            )
            for f_in, f_out in zip(
                self.att_filters[:0:-1], self.att_filters[-2::-1]
            )
        ])
        self.final = MultiheadedAttention(self.att_filters[0], channels)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(
                    p, t.type_as(p).to(p.device)
                )
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        for e_tf in self.encoder:
            e_tf.to(self.device)
            data = e_tf(data)
        for d_tf in self.decoder:
            d_tf.to(self.device)
            data = d_tf(data)
        self.final.to(self.device)
        return self.final(data)

    def next_motif(self, motif):
        tensor_motif = torch.from_numpy(
            np.expand_dims(motif, axis=0)
        ).to(self.device)
        next_motif = torch.sigmoid(self(tensor_motif))

        return next_motif[0].cpu().numpy()

    def song(self, motif, n_motifs):
        song_list = [motif]
        for _ in range(n_motifs):
            motif = self.next_motif(motif)
            song_list.append(motif)

        return np.concatenate(song_list, axis=1)


class MultiheadedAttention(nn.Module):
    """
        Mmulti-headed attention based on
        A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, Ll. Jones, A.N. Gomez,
        L. Kaiser, I. Polosukhin
        "Attention Is All You Need"
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self, in_features, att_features, heads=8,
            norm=partial(torch.softmax, dim=1),
    ):
        super().__init__()
        assert att_features % heads == 0,\
            'The number of attention features must be divisible by ' \
            'the number of blocks'
        self.blocks = heads
        self.out_features = att_features
        self.features = att_features // heads
        self.sa_blocks = nn.ModuleList([
            SelfAttention(
                in_features, self.features, norm
            )
            for _ in range(self.blocks)
        ])
        self.final_block = nn.Sequential(
            nn.Conv1d(att_features, att_features, 1),
            nn.ReLU(att_features),
            nn.Conv1d(att_features, att_features, 1)
        )

    def forward(self, x):
        x = torch.cat([sa_i(x) for sa_i in self.sa_blocks], dim=1)
        z = self.final_block(x)
        return z


class SelfAttention(nn.Module):
    """
        Non-local self-attention block based on
        X. Wang, R. Girshick, A.Gupta, K. He
        "Non-local Neural Networks"
        https://arxiv.org/abs/1711.07971
    """

    def __init__(
            self, in_features, att_features,
            norm=partial(torch.softmax, dim=1)
    ):
        super().__init__()
        self.features = att_features
        self.map_key = nn.Conv1d(in_features, att_features, 1)
        self.map_query = nn.Conv1d(in_features, att_features, 1)
        self.map_value = nn.Conv1d(in_features, att_features, 1)
        self.norm = norm

    def forward(self, x):
        key = self.map_key(x)
        query = self.map_query(x)
        value = self.map_value(x)

        att = torch.bmm(key.transpose(1, 2), query)
        att_map = self.norm(
            att.flatten(1) / np.sqrt(self.features)
        ).view_as(att)
        self_att = torch.bmm(value, att_map)

        return self_att
