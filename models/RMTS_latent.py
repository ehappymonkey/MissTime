# 使用available channels在线检索。
# 配合onlineRetrieval使用，目前效果不如不检索的Dlinear。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from layers.onlineRetrieval import RetrievalTool, Retrieval
from layers.latentRetrieval import LatentRetrieval

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        print('Initializing RMTS_latent Model...')
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        self.rt = LatentRetrieval(
            topk=self.topm,
            )
        self.ts2vec_params={'output_dims': configs.latent_dim, 'batch_size':configs.batch_size} # ts2vec需要自己建立data_loader来训练。
        
        # args
        self.args = configs
        self.encoder = None
 
        self.num_classes = getattr(configs, 'num_class', 10)  # 分类任务类别数

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.linear_x = nn.Linear(self.seq_len, self.pred_len)
            self.linear_x2 = nn.Linear(320, self.pred_len*self.channels)
        elif self.task_name == 'classification':
            self.classifier = nn.Linear(320, self.num_classes)
        
        self.repr_decoder = nn.Linear(320, self.channels)  # TS2Vec output_dims=320
        
    def prepare_dataset(self, train_data, task_name="classification"):
        """
        准备检索用的数据集
        Args:
            train_data: (N, L, C) 训练数据集
            task_name: "classification" or "forecasting"
            处理数据集+训练encoder。
        """
        train_x_list = []
        for i in range(len(train_data)):
            if task_name == 'classification' or task_name == 'anomaly_detection':
                td = train_data[i][0]
            else:
                td = train_data[i][1]
            train_x_list.append(td)  # td: (seq_len, channels)
        train_data = torch.tensor(np.stack(train_x_list, axis=0)).float()  # (N, L, C)
        
        from layers.ts2vec import TS2Vec
        print("Training TS2Vec encoder for Latent Retrieval...")
        encoder = TS2Vec(input_dims=self.channels, **self.ts2vec_params)
        encoder.fit(train_data.cpu().numpy(), verbose=True)
        print("TS2Vec encoder trained.")
        self.encoder = encoder
        self.rt.prepare_dataset(train_data, encoder, task_name)
        

    def forward(self, x_enc, index, batch_mask, x_for_retrieval=None, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, batch_mask, x_for_retrieval, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, batch_mask, x_for_retrieval, mode)
            return dec_out  # [B, N]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, batch_mask, x_for_retrieval, mode)
            return dec_out
        return None


    def forecast(self, x_enc, index, batch_mask, x_for_retrieval=None, mode='train'):
        """
        预测任务：用检索填充缺失，然后线性预测
        """
        x = x_enc  # (B, L, C)
        # 这里的x_for_retrieval有维度问题，！= x?
        # print("x shape", x.shape)
        # print("x_retrieval", x_for_retrieval.shape)
        # 测试时用 retrieval 填充
        import time
        begin_time = time.time()
        
        if mode != 'train': # 用重建的表示做预测
            if self.args.use_full_retrieval:
                # Oracle mode: 用完整 x 做 retrieval
                x_recon_repr = self.rt.retrieve_recon(
                    x_for_retrieval, 
                    observed_mask=torch.ones_like(batch_mask),
                    task_name="forecasting"
                )
            elif self.args.use_no_retrieval:
                # 测试时也不用检索，直接用缺失的x编码
                x_recon_repr = self.encoder.encode(
                    x, 
                    encoding_window=None
                )  # (B, L, D)
            else:
                # Normal mode: 用 masked的 x 做 retrieval
                x_recon_repr = self.rt.retrieve_recon(
                    x, 
                    observed_mask=batch_mask,
                    task_name="forecasting"
                )  # (B, L, D)
            x_repr = x_recon_repr
        else: # 训练时用自己的完整表示做预测
            x_repr = self.encoder.encode(x, encoding_window=None)  # (B, L, D)
            x_repr = torch.tensor(x_repr).float().to(x.device)
            
        # x_pepr = x_repr[:, -1, :]
        # x_pred = self.linear_x2(x_pepr)  # (B, pred_len*channels)
        # x_pred = x_pred.reshape(x.shape[0], self.pred_len, self.channels)
        # return x_pred

        x = self.repr_decoder(x_repr)  # (B, L, C)
        end_time = time.time()
        # print(f"Retrieval time: {end_time - begin_time} seconds")
        x_pred = self.linear_x(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, C)
        return x_pred  # (B, pred_len, C)

    def classification(self, x_enc, batch_mask, x_for_retrieval=None, mode='train'):
        """
        分类任务：用检索得到完整序列表示，接分类头
        """
        x = x_enc  # (B, L, C)
        
        if mode != 'train':
            if self.args.use_full_retrieval:
                repr = self.rt.retrieve_recon(
                    x_for_retrieval,
                    observed_mask=torch.ones_like(batch_mask),
                    task_name="classification"
                )
            elif self.args.use_no_retrieval:
                # 测试时也不用检索，直接用缺失的x编码
                repr = self.encoder.encode(
                    x, 
                    encoding_window='full_series'
                )  # (B, D)
            else: # 测试时用 masked 的 x 做 retrieval
                repr = self.rt.retrieve_recon(
                    x,
                    observed_mask=batch_mask,
                    task_name="classification"
                )  # (B, D)
        else:
            # 训练时用完整 x（假设无缺失）编码
            repr = self.encoder.encode(
                x,
                encoding_window='full_series'
            )
            # repr = torch.tensor(repr).float().to(x.device)  # (B, D)
        
        # 分类头（需在 __init__ 中定义）
        logits = self.classifier(repr)  # (B, num_classes)
        return logits

    def anomaly_detection(self, x_enc, batch_mask, x_for_retrieval=None, mode='train'):
        """
        异常检测：返回重构误差（与原始序列比较）
        """
        x = x_enc  # (B, L, C)
        
        if mode != 'train':
            if x_for_retrieval is not None:
                repr = self.rt.retrieve_recon(
                    x_for_retrieval,
                    observed_mask=torch.ones_like(batch_mask).bool(),
                    task_name="anomaly_detection"
                )
            else:
                repr = self.rt.retrieve_recon(
                    x,
                    observed_mask=batch_mask.bool(),
                    task_name="anomaly_detection"
                )  # (B, L, D)
            x_recon = self.repr_to_raw(repr)  # (B, L, C)
        else:
            # 训练时：直接重构
            x_recon = self.repr_to_raw(
                self.rt.encoder.encode(x.cpu().numpy(), encoding_window=None)
            )
            x_recon = torch.tensor(x_recon).float().to(x.device)
        
        # 返回重构序列（用于计算 MSE）
        return x_recon

    def repr_to_raw(self, repr):
        """
        将 TS2Vec 表示映射回原始空间 (B, *, D) -> (B, *, C)
        """
        if repr.dim() == 3:  # (B, L, D)
            B, L, D = repr.shape
            repr_flat = repr.reshape(-1, D)
            raw_flat = self.repr_decoder(repr_flat)
            return raw_flat.reshape(B, L, self.channels)
        else:  # (B, D)
            return self.repr_decoder(repr)  # (B, C)，但分类任务不需要