from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from base import BaseModel



def accuracy_logits(logits, target):
    prediction = torch.max(logits, dim=1)[1]
    return 1 - (prediction == target).float().mean()
    
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim, heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Linear(embed_dim, mlp_dim)

    def forward(self, x_in, mask=None):
        x = self.ln1(x_in)
        print(x)
        x, _ = self.attention(
            query=x, key=x, value=x, attn_mask=mask,
            need_weights=False,
        )
        x = x + x_in

        y = self.ln2(x)
        y = self.mlp(y)

        return x + y
    

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
        encoder_depth=16,
        decoder_depth=16,
        device=torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        notes=128,
        bits=8,
        multitokens=True,
        heads=32,
        lr=1e-3,
        verbose=0,
    ):
        super().__init__()
        # Init values
        self.epoch = None
        self.t_train = 0
        self.t_val = 0
        self.heads = heads
        self.device = device
        self.multitokens = multitokens
        if self.multitokens:            
            channels = notes
        else:
            channels = 2 * notes + bits

        # <Parameter setup>
        self.encoder = nn.ModuleList([
            SelfAttentionBlock(channels, channels, heads)
            for _ in range(encoder_depth)
        ])
        self.decoder = nn.ModuleList([
            SelfAttentionBlock(channels, channels, heads)
            for _ in range(decoder_depth)
        ])

        # <Loss function setup>
        if self.multitokens:
            self.train_functions = [
                {
                    'name': 'xent',
                    'weight': 1,
                    'f': F.binary_cross_entropy_with_logits
                },
            ]
            
            self.val_functions = [
                {
                    'name': 'xent',
                    'weight': 1,
                    'f': F.binary_cross_entropy_with_logits
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
        else:
            self.train_functions = [
                {
                    'name': 'xent',
                    'weight': 1,
                    'f': F.cross_entropy
                },
            ]
            
            self.val_functions = [
                {
                    'name': 'xent',
                    'weight': 0,
                    'f': F.cross_entropy
                },
                {
                    'name': 'acc',
                    'weight': 1,
                    'f': accuracy_logits
                },
            ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        # self.optimizer_alg = torch.optim.Adam(model_params, lr=lr)
        self.optimizer_alg = torch.optim.SGD(model_params, lr=lr)
        self.schedulers = [
#             torch.optim.lr_scheduler.ExponentialLR(
#                 self.optimizer_alg, gamma=0.9
#             ),
#             torch.optim.lr_scheduler.MultiStepLR(
#                 self.optimizer_alg, milestones=[10, 30, 50, 80, 100, 150], gamma=0.1
#             ),
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_alg, 'min'
            )
        ]
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
        N, F, L = data.shape
        mask = torch.ones((L, L), dtype=bool)
        mask = torch.logical_not(torch.triu(mask)).to(self.device)
        seq_range = torch.arange(0, data.shape[-1])
        x_cord, y_cord = torch.meshgrid(seq_range, seq_range)
        s_rel = 1 - torch.abs(x_cord - y_cord).type_as(data).to(data.device)
        snorm_rel = s_rel / L
        data = data.transpose(-1, -2)
        for i, e_tf in enumerate(self.encoder):
            e_tf.to(self.device)
            data = e_tf(data, snorm_rel)
        for i, d_tf in enumerate(self.decoder):
            d_tf.to(self.device)
            data = d_tf(data, mask)
        return data.transpose(-1, -2)

    def next_beat(self, motif):
        self.eval()
        with torch.no_grad():
            tensor_motif = torch.from_numpy(
                np.expand_dims(motif, axis=0)
            ).to(self.device)
            if self.multitokens:
                next_beat = torch.sigmoid(self(tensor_motif))
            else:
                next_beat = torch.softmax(self(tensor_motif), dim=1)

        return next_beat.detach().cpu().numpy()[0, ...]

    def song(self, motif, n_beats):
        song_list = [motif]
        song = [motif]
        for _ in range(n_beats):
            beat = self.next_beat(motif)
            new_notes = deepcopy(beat)
            if self.multitokens:
                motif = (new_notes > 0.5).astype(np.float32)
                song_list.append(
                    beat
                )
                song.append(
                    new_notes > 0.5
                )
            else:
                new_tokens = deepcopy(beat)
                max_val = np.max(new_tokens, axis=0, keepdims=True)
                motif = (new_tokens == max_val).astype(np.float32)
                song_list.append(
                    beat
                )
                song.append(
                    new_tokens == max_val
                )

        return np.concatenate(song_list, axis=1), np.concatenate(song, axis=1)



class MonophonicTransformer(BaseModel):
    def __init__(
        self,
        encoder_depth=16,
        decoder_depth=16,
        device=torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        heads=32,
        lr=1e-3,
        verbose=0,
    ):
        super().__init__()
        # Init values
        self.epoch = None
        self.t_train = 0
        self.t_val = 0
        self.heads = heads
        self.device = device

        # <Parameter setup>
        self.encoder = nn.ModuleList([
            SelfAttentionBlock(16, 1, heads)
            for _ in range(encoder_depth)
        ])
        self.decoder = nn.ModuleList([
            SelfAttentionBlock(16, 1, heads)
            for _ in range(decoder_depth)
        ])

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': F.cross_entropy
            },
        ]
        
        self.val_functions = [
            {
                'name': 'xent',
                'weight': 0,
                'f': F.cross_entropy
            },
            {
                'name': 'acc',
                'weight': 1,
                'f': accuracy_logits
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        # self.optimizer_alg = torch.optim.Adam(model_params, lr=lr)
        self.optimizer_alg = torch.optim.SGD(model_params, lr=lr)
        self.schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_alg, 'min'
            )
        ]
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
        N, L = data.shape
        mask = torch.ones((L,), dtype=bool)
        # mask = torch.logical_not(torch.triu(mask)).to(self.device)
        seq_range = torch.arange(0, data.shape[-1])
        x_cord = seq_range
        s_rel = 1 - torch.abs(x_cord).type_as(data).to(data.device)
        snorm_rel = s_rel / L
        data = torch.tensor(data.transpose(0, 1), dtype=int)
        for i, e_tf in enumerate(self.encoder):
            e_tf.to(self.device)
            data = e_tf(data, snorm_rel)
        for i, d_tf in enumerate(self.decoder):
            d_tf.to(self.device)
            data = d_tf(data, mask)
        return data.transpose(0, 1)

    def next_beat(self, motif):
        self.eval()
        with torch.no_grad():
            tensor_motif = torch.from_numpy(
                np.expand_dims(motif, axis=0)
            ).to(self.device)
            if self.multitokens:
                next_beat = torch.sigmoid(self(tensor_motif))
            else:
                next_beat = torch.softmax(self(tensor_motif), dim=1)

        return next_beat.detach().cpu().numpy()[0, ...]

    def song(self, motif, n_beats):
        song_list = [motif]
        song = [motif]
        for _ in range(n_beats):
            beat = self.next_beat(motif)
            new_notes = deepcopy(beat)
            if self.multitokens:
                motif = (new_notes > 0.5).astype(np.float32)
                song_list.append(
                    beat
                )
                song.append(
                    new_notes > 0.5
                )
            else:
                new_tokens = deepcopy(beat)
                max_val = np.max(new_tokens, axis=0, keepdims=True)
                motif = (new_tokens == max_val).astype(np.float32)
                song_list.append(
                    beat
                )
                song.append(
                    new_tokens == max_val
                )

        return np.concatenate(song_list, axis=1), np.concatenate(song, axis=1)

























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
