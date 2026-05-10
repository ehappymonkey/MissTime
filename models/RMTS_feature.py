


import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.onlineRetrieval import RetrievalTool, Retrieval


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        print('Initializing RAFT Model...')
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
#         self.decompsition = series_decomp(configs.moving_avg)
#         self.individual = individual
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        # self.rt = RetrievalTool(
        #     seq_len=self.seq_len,
        #     pred_len=self.pred_len,
        #     channels=self.channels,
        #     n_period=self.n_period,
        #     topm=self.topm,
        # )
        # self.period_num = self.rt.period_num[-1 * self.n_period:]

        self.rt = Retrieval(
            topk=self.topm,
            temperature=0.1,
        )
        
        # module_list = [

        #     for g in self.period_num
        # ]
        # self.retrieval_recon = nn.ModuleList(module_list)
        # # self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        # args
        self.args = configs

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)




    def prepare_dataset(self, train_data, task_name=None):
        self.rt.prepare_dataset(train_data, task_name)
        



    def encoder(self, x, index, batch_mask, x_full, mode):

        
        bsz, seq_len, channels = x.shape
        # assert(seq_len == self.seq_len, channels == self.channels)



        # x_recon = self.rt.retrieve_recon(x, index, observed_mask=batch_mask, train=mode) # G, batch, seq_len, channel
        
        if mode != 'train':
            if self.args.use_full_retrieval:

                full_obs_mask = torch.ones_like(batch_mask).to(batch_mask.device)  # (C,)
                x_recon = self.rt.retrieve_recon(x_full, observed_mask=full_obs_mask)
            else:

                x_recon = self.rt.retrieve_recon(x, observed_mask=batch_mask) # G, batch, seq_len, channel
            x = torch.where(batch_mask.bool().to(x.device), x, x_recon)

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1) # B, P, C
         
        pred = x_pred_from_x
        pred = pred + x_offset
        
        return pred
    

    def classification(self, x_enc, batch_mask, x_for_retrieval, mode):
        # Encoder
        enc_out = self.encoder(x_enc, None, batch_mask, x_for_retrieval, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forecast(self, x_enc, index, batch_mask, x_for_retrieval, mode):
        # Encoder
        return self.encoder(x_enc, index, batch_mask, x_for_retrieval, mode)
    
    def anomaly_detection(self, x_enc, batch_mask, x_for_retrieval, mode):
        # Encoder
        return self.encoder(x_enc, None, batch_mask, x_for_retrieval, mode)

    def forward(self, x_enc, index, batch_mask, x_for_retrieval=None, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, batch_mask,x_for_retrieval, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, batch_mask, x_for_retrieval, mode)
            return dec_out  # [B, N]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, batch_mask, x_for_retrieval, mode)
            return dec_out
        return None
