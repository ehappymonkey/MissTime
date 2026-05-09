import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Embed import PositionalEmbedding
from models.TimeFilter_utils import TimeFilter_Backbone

import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    

class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)
    
    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x) # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1) # L

        # Filter
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # embed
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        # TimeFilter Backbone
        self.backbone = TimeFilter_Backbone(self.dim, self.n_vars, self.d_ff,
                                  configs.n_heads, configs.e_layers, self.top_p, configs.dropout, self.seq_len * self.n_vars // self.patch_len)
        
        if self.task_name == 'imputation':
            # 插补任务：重建整个输入序列
            self.head = nn.Linear(self.dim * self.num_patches, self.seq_len)
        else:
            # 预测任务：预测未来 pred_len
            self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

   

        # Without RevIN
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)
        self.args = configs
        self.device = torch.device('cuda:{}'.format(self.args.gpu))
        self.masks = self._get_mask()

    def _get_mask(self):
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks
    
    def forward(self, x, is_training=False):
        masks  = self.masks
        # x: [B, T, C]
        B, T, C = x.shape
        # Normalization
        x = self.norm(x, 'norm')
        # x: [B, C, T]
        x = x.permute(0, 2, 1).reshape(-1, C*T) # [B, C*T]
        x = self.patch_embed(x) # [B, N, D]  N = [C*T / P]

        x, moe_loss = self.backbone(x, masks, self.alpha, is_training)


        if self.task_name == 'imputation':
            # 重建整个输入序列 (T = seq_len)
            x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))  # [B, C, T]
            x = x.permute(0, 2, 1)  # [B, T, C]
        else: 
            # Forecasting
            # [B, C, T/P, D]
            x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2)) # [B, C, T]
            x = x.permute(0, 2, 1)
            
        
        
        
        # De-Normalization
        x = self.norm(x, 'denorm')

        return x, moe_loss
        