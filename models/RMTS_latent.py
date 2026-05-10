


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
        self.ts2vec_params={'output_dims': configs.latent_dim, 'batch_size':configs.batch_size}
        
        # args
        self.args = configs
        self.encoder = None
 
        self.num_classes = getattr(configs, 'num_class', 10)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.linear_x = nn.Linear(self.seq_len, self.pred_len)
            self.linear_x2 = nn.Linear(320, self.pred_len*self.channels)
        elif self.task_name == 'classification':
            self.classifier = nn.Linear(320, self.num_classes)
        
        self.repr_decoder = nn.Linear(320, self.channels)  # TS2Vec output_dims=320
        
    def prepare_dataset(self, train_data, task_name="classification"):
        """
        
        Args:
            train_data: (N, L, C) 
            task_name: "classification" or "forecasting"
            +encoder
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
        ：，
        """
        x = x_enc  # (B, L, C)

        # print("x shape", x.shape)
        # print("x_retrieval", x_for_retrieval.shape)

        import time
        begin_time = time.time()
        
        if mode != 'train':
            if self.args.use_full_retrieval:

                x_recon_repr = self.rt.retrieve_recon(
                    x_for_retrieval, 
                    observed_mask=torch.ones_like(batch_mask),
                    task_name="forecasting"
                )
            elif self.args.use_no_retrieval:

                x_recon_repr = self.encoder.encode(
                    x, 
                    encoding_window=None
                )  # (B, L, D)
            else:

                x_recon_repr = self.rt.retrieve_recon(
                    x, 
                    observed_mask=batch_mask,
                    task_name="forecasting"
                )  # (B, L, D)
            x_repr = x_recon_repr
        else:
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
        ：，
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

                repr = self.encoder.encode(
                    x, 
                    encoding_window='full_series'
                )  # (B, D)
            else:
                repr = self.rt.retrieve_recon(
                    x,
                    observed_mask=batch_mask,
                    task_name="classification"
                )  # (B, D)
        else:

            repr = self.encoder.encode(
                x,
                encoding_window='full_series'
            )
            # repr = torch.tensor(repr).float().to(x.device)  # (B, D)
        

        logits = self.classifier(repr)  # (B, num_classes)
        return logits

    def anomaly_detection(self, x_enc, batch_mask, x_for_retrieval=None, mode='train'):
        """
        ：（）
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

            x_recon = self.repr_to_raw(
                self.rt.encoder.encode(x.cpu().numpy(), encoding_window=None)
            )
            x_recon = torch.tensor(x_recon).float().to(x.device)
        

        return x_recon

    def repr_to_raw(self, repr):
        """
         TS2Vec  (B, *, D) -> (B, *, C)
        """
        if repr.dim() == 3:  # (B, L, D)
            B, L, D = repr.shape
            repr_flat = repr.reshape(-1, D)
            raw_flat = self.repr_decoder(repr_flat)
            return raw_flat.reshape(B, L, self.channels)
        else:  # (B, D)
            return self.repr_decoder(repr)