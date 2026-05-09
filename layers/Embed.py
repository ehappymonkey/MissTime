import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x) # x: B, L, 1


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model) # c_in指的是seq_len。
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, mask=None):
        x = x.permute(0, 2, 1)
        
        # # x: [Batch Variate Time]； x_mark: [Batch, 4, Time]
        # if x_mark is None:
        #     x = self.value_embedding(x)
        # else:
        #     x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate(+4) d_model]

        x = self.value_embedding(x) # 只使用本身embedding。
        return self.dropout(x)

class TopologyAwareEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 假设 c_in 是变量数量 (C)
        self.c_in = configs.enc_in 
        self.latent_dim = configs.latent_dim
        
        self.data_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.latent_dim, configs.embed, configs.freq, configs.dropout
        )
        
        # 这里改+mark维度
        self.variate_embedding = nn.Parameter(torch.randn(1, self.c_in, self.latent_dim))
        # self.variate_embedding = nn.Parameter(torch.randn(1, self.c_in+configs.dim_x_mark, self.latent_dim))
        # 0: Observed, 1: Missing
        self.mask_embedding = nn.Embedding(2, self.latent_dim)
        
    def forward(self, x, x_mark, mask):
        """
        x: [B, L, C] (缺失处已填 0)
        mask: [B, C] (1: 存在, 0: 缺失) -> 注意这里定义反一下方便 embedding
        """
        
        # Step 1: Content Embedding
        # [B, L, C] -> [B, C, D]
        x_enc = self.data_embedding(x, x_mark) # B, L, C/ B, L, 4, 这里用iTransformer编码相当于多了4个变量（token）。Variate ID编码并没有考虑这几个时间token。
        
        # Step 2: Add Variate Embedding (Broadcasting)
        # [B, C, D] + [1, C, D] -> [B, C, D]
        x_enc = x_enc + self.variate_embedding
        
        # Step 3: Add Mask Embedding
        # mask 输入是 [B, C], 1代表存在, 0代表缺失
        # 为了配合 Embedding(2, D)，我们需要把 mask 变成 index
        # 假设我们定义: embedding(0)=Observed_Emb, embedding(1)=Missing_Emb
        # 那么我们需要把输入的 mask (1存在) 变成 (0存在), (0缺失) 变成 (1缺失)

        # d_mark = x_mark.shape[-1] if x_mark is not None else 0

        # if d_mark > 0:
        #     mark_mask = torch.ones(x.shape[0], d_mark, device=x.device)
        #     # [B, C] + [B, D_mark] -> [B, C + D_mark]
        #     full_mask = torch.cat([mask, mark_mask], dim=1)
        # else:
        #     full_mask = mask

        # mask_idx = 1 - full_mask.long() # [B, C]
        # [B, C, D]
        # 避免x_mark影响。
        mask_idx = 1 - mask.long()
        mask_emb = self.mask_embedding(mask_idx)
        x_enc = x_enc + mask_emb
        return x_enc

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars