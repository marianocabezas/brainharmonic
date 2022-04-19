from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from base import BaseModel


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    """
    Function to compute the focal loss based on:
    Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. "Focal
    Loss for Dense Object Detection".
    https://arxiv.org/abs/1708.02002
    https://ieeexplore.ieee.org/document/8237586
    :param pred: Predicted values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param target: Ground truth values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param alpha: Weighting parameter to avoid class imbalance (default 0.2).
    :param gamma: Focusing parameter (default 2.0).
    :return: Focal loss value.
    """

    m_bg = target == 0
    m_fg = target > 0

    if alpha is None:
        alpha = float(torch.sum(m_bg)) / torch.numel(target)

    alpha_fg = alpha
    alpha_bg = 1 - alpha
    pt_fg = pred[m_fg]
    pt_bg = (1 - pred[m_bg])

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    bce_fg = bce[m_fg]
    bce_bg = bce[m_bg]

    focal_fg = alpha_fg * (1 - pt_fg).pow(gamma) * bce_fg
    focal_bg = alpha_bg * (1 - pt_bg).pow(gamma) * bce_bg

    focal = torch.cat([focal_fg, focal_bg])

    return focal.mean()


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
        # Init values
        if att_filters is None:
            self.att_filters = [64, 32, 16]
        else:
            self.att_filters = att_filters
        self.epoch = None
        self.t_train = 0
        self.t_val = 0
        self.device = device
        encoder_channels = [channels] + self.att_filters

        # <Parameter setup>
        self.encoder = nn.ModuleList([
            nn.Sequential(
                MultiheadedAttention(f_in, f_out, f_in // 8),
                nn.ReLU(),
                nn.GroupNorm(f_out // 8, f_out),
            )
            for f_in, f_out in zip(
                encoder_channels[:-1], encoder_channels[1:]
            )
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(
                MultiheadedAttention(f_in, f_out, f_in // 4),
                nn.ReLU(),
                nn.GroupNorm(f_out // 4, f_out),
            )
            for f_in, f_out in zip(
                self.att_filters[:0:-1], self.att_filters[-2::-1]
            )
        ])
        self.final = MultiheadedAttention(
            self.att_filters[0], channels, self.att_filters[0] // 2
        )

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xent',
                'weight': 1,
                # 'f': lambda p, t: F.binary_cross_entropy_with_logits(
                'f': lambda p, t: focal_loss(
                    p, t,
                )
            },
            {
                'name': 'l1',
                'weight': 1,
                'f': lambda p, t: F.l1_loss(
                    torch.sigmoid(p), t,
                )
            },
            # {
            #     'name': 'bent',
            #     'weight': 1,
            #     'f': lambda p, t: torch.sum(
            #         - torch.softmax(p, dim=1) * torch.log(torch.softmax(p, dim=1)),
            #         dim=0
            #     )
            # }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(
                    p, t,
                )
            },
            {
                'name': 'mse',
                'weight': 0,
                'f': lambda p, t: F.mse_loss(
                    torch.sigmoid(p), t,
                )
            },
            {
                'name': 'l1',
                'weight': 0,
                'f': lambda p, t: F.l1_loss(
                    torch.sigmoid(p), t,
                )
            },
            {
                'name': '0mse',
                'weight': 0,
                'f': lambda p, t: F.mse_loss(
                    torch.zeros_like(t).to(t.device), t,
                )
            },
            {
                'name': '1mse',
                'weight': 0,
                'f': lambda p, t: F.mse_loss(
                    torch.ones_like(t).to(t.device), t,
                )
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.AdamW(model_params, lr=1e-4)
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
        data = self.final(data)
        return data[:, :, -1:]

    def next_beat(self, motif):
        self.eval()
        with torch.no_grad():
            tensor_motif = torch.from_numpy(
                np.expand_dims(motif, axis=0)
            ).to(self.device)
            next_beat = torch.sigmoid(self(tensor_motif))
            np_beat = next_beat[-1].detach().cpu().numpy()

        return np_beat

    def song(self, motif, n_beats):
        song_list = [motif]
        for _ in range(n_beats):
            note = self.next_beat(motif)
            new_note = deepcopy(note)
            # new_note[np.argsort(note, axis=0)[:-6], :] = 0
            # new_note[np.argsort(note, axis=0)[:-7:-1], :] = 1
            motif = np.concatenate([motif, new_note > 0.5], axis=-1)
            # song_list.append(np.exp(note) / sum(np.exp(note)))
            note_sum = np.sum(np.max(note))
            norm_note = note / note_sum if note_sum > 0 else note
            song_list.append(
                # norm_note
                note
            )

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
            self, in_features, att_features, heads=32,
            norm=partial(torch.softmax, dim=-1),
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
            nn.ReLU(),
            nn.GroupNorm(heads, self.features),
            nn.Conv1d(att_features, att_features, 1),
            nn.ReLU(),
            nn.GroupNorm(heads, self.features),
            nn.Conv1d(att_features, att_features, 1)
        )
        self.final_block = nn.Conv1d(att_features, att_features, 1)

    def forward(self, x):
        x = torch.cat([sa_i(x) for sa_i in self.sa_blocks], dim=1)
        z = self.final_block(x) + x
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
            norm=partial(torch.softmax, dim=2)
    ):
        super().__init__()
        self.features = att_features
        self.map_key = nn.Conv1d(in_features, att_features, 1)
        self.map_query = nn.Conv1d(in_features, att_features, 1)
        self.map_value = nn.Conv1d(in_features, att_features, 1)
        self.norm = norm

    def forward(self, x):
        key = F.instance_norm(self.map_key(x))
        # key = self.map_key(x)
        query = F.instance_norm(self.map_query(x))
        # query = self.map_query(x)
        value = self.map_value(x)

        # seq_range = torch.arange(0, x.shape[-1])
        att = torch.bmm(query.transpose(1, 2), key)
        # x_cord, y_cord = torch.meshgrid(seq_range, seq_range)
        # s_rel = 1 - torch.abs(x_cord - y_cord).type_as(x).to(x.device)
        # snorm_rel = s_rel / x.shape[-1]
        # att_map = self.norm((att * snorm_rel) / np.sqrt(self.features))
        masked_att = torch.triu(att)
        # att_map = self.norm(att / np.sqrt(self.features))
        att_map = self.norm(masked_att / np.sqrt(self.features))

        self_att = torch.bmm(value, att_map)

        return self_att
