# 这个TimesNet结合的是iTransformer检索器。


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import time

from rag.miss_retrieval_cpu import missRetrieval
from rag.feat_retrieval import Retrieval


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x.float(), dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

        self.device = torch.device('cuda:{}'.format(configs.gpu)) if configs.use_gpu else torch.device('cpu')
        if configs.rag_type == 'latent_rag':
            self.rt = missRetrieval(
                configs, 
                topk=configs.topm,
                temperature=0.1,
                device=self.device,
                retriever=configs.retrieve_encoder, # default iTransformer
            )
        elif configs.rag_type == 'feature_rag':
            self.rt = Retrieval(
                configs,
                topk=configs.topm,
                temperature=0.1,
                device=self.device,
                retriever='RawFeature'
            )

        self.args = configs
        self.time_encoding = 0
        self.time_training = 0
        self.time_inf = 0
        self.time_rag = 0
        
    def prepare_retrieval(self, train_loader_contra, vali_loader_contra, train_loader_unshuffled):
        enc0 = time.time()
        # 对比学习训练Retrieval Encoder。
        if self.args.rag_type == 'latent_rag':
            self.rt.training_encoder(self.args, train_loader_contra, vali_loader_contra)
            # Encode训练集的embedding并存储。
            self.rt.prepare_dataset(self.args, train_loader_unshuffled) 
        # self.rt.load_kb_to_gpu()  # 存储张量移到GPU里。
        elif self.args.rag_type == 'feature_rag':
            self.rt.prepare_dataset(self.args, train_loader_unshuffled)
        enc1 = time.time()
        self.time_encoding += (enc1 - enc0)
        print(f'Encoding time for retrieval: {self.time_encoding:.2f} seconds')

    def forecast(self, x_enc, x_mark_enc, batch_mask, x_full, mode):

        # feature-based RAG 补全缺失的通道。
        if mode != 'train':
            if mode == 'test':
                rag_0 = time.time()
            if self.args.rag_type == 'latent_rag':
                # Normal mode: 用 masked x 的observed variables检索
                batch_mask = batch_mask.unsqueeze(0).expand(x_enc.shape[0], -1)
                if self.args.rag_strategy == 'channels':
                    x_recon = self.rt.retrieve_recon(x_enc, x_mark_enc, batch_mask) # batch, seq_len, channel
                elif self.args.rag_strategy == 'whole':
                    x_recon = self.rt.retrieve_recon_whole(x_enc, x_mark_enc, batch_mask)
                x_enc = torch.where(batch_mask.unsqueeze(1).bool().to(x_enc.device), x_enc, x_recon)
            elif self.args.rag_type == 'feature_rag':
                x_recon = self.rt.retrieve_recon(x_enc)
            elif self.args.rag_type == 'no_rag':
                # 测试时也不用检索，直接用缺失的x编码
                x_recon = x_enc   
            if mode == 'test':
                rag_1 = time.time()
                self.time_rag += (rag_1 - rag_0)

        if mode == 'test':
            inf_0 = time.time()
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))

        if mode == 'test':
            inf_1 = time.time()
            self.time_inf += (inf_1 - inf_0)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, batch_mask, x_full, mode, mask=None):
        if mode != 'train':
            if self.args.rag_type == 'latent_rag':
                # Normal mode: 用 masked x 的observed variables检索
                batch_mask = batch_mask.unsqueeze(0).expand(x_enc.shape[0], -1)
                x_recon = self.rt.retrieve_recon(x_enc, x_mark_enc, batch_mask) # batch, seq_len, channel
                x_enc = torch.where(batch_mask.unsqueeze(1).bool().to(x_enc.device), x_enc, x_recon)
            elif self.args.rag_type == 'no_rag':
                # 测试时也不用检索，直接用缺失的x编码
                x_recon = x_enc   
            
        
        if mask is None:
            mask = torch.ones_like(x_enc).to(x_enc.device)
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum((mask == 1).float(), dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out# , x_rag_recon

    def anomaly_detection(self, x_enc, batch_mask, x_full, mode):
        x_recon_feat = x_enc
        # feature-based RAG 补全缺失的通道。
        if mode != 'train':
            if self.args.use_full_retrieval:
                # Oracle mode: 用完整 x 检索
                full_obs_mask = torch.ones_like(batch_mask).to(batch_mask.device)  # (C,)
                x_recon = self.rt.retrieve_recon(x_full, observed_mask=full_obs_mask)
            elif self.args.rag_type == 'feature_rag' or self.args.rag_type == 'latent_rag':
                # Normal mode: 用 masked x 的observed variables检索
                x_recon = self.rt.retrieve_recon(x_enc, observed_mask=batch_mask) # batch, seq_len, channel
            else:
                # 测试时也不用检索，直接用缺失的x编码
                x_recon = x_enc
            x_enc = torch.where(batch_mask.bool().to(x_enc.device), x_enc, x_recon)
            x_recon_feat = x_enc

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out, x_recon_feat # 这里直接使用了RAG的检索结果作为分数计算，没有经过模型？

    def classification(self, x_enc, batch_mask, x_full, mode):
        # feature-based RAG 补全缺失的通道。
        if mode != 'train':
            if self.args.use_full_retrieval:
                # Oracle mode: 用完整 x 检索
                full_obs_mask = torch.ones_like(batch_mask).to(batch_mask.device)  # (C,)
                x_recon = self.rt.retrieve_recon(x_full, observed_mask=full_obs_mask)
            elif self.args.rag_type == 'feature_rag' or self.args.rag_type == 'latent_rag':
                # Normal mode: 用 masked x 的observed variables检索
                x_recon = self.rt.retrieve_recon(x_enc, observed_mask=batch_mask) # batch, seq_len, channel
            else:
                # 测试时也不用检索，直接用缺失的x编码
                x_recon = x_enc
            x_enc = torch.where(batch_mask.bool().to(x_enc.device), x_enc, x_recon)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_mask, batch_x_full, mode, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, batch_mask, batch_x_full, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, batch_mask, batch_x_full, mode, mask)
            return dec_out # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out, x_recon_feat = self.anomaly_detection(x_enc, batch_mask, batch_x_full, mode)
            return dec_out, x_recon_feat  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, batch_mask, batch_x_full, mode)
            return dec_out  # [B, N]

        return None