import numpy as np
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(
                f"Salesforce/moirai-2.0-R-small",
            ),
            prediction_length=configs.pred_len,
            context_length=configs.seq_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to('cuda')

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if True:
            if self.args.rag_type == 'latent_rag':

                batch_mask = batch_mask.unsqueeze(0).expand(x_enc.shape[0], -1)
                if self.args.rag_strategy == 'channels':
                    x_recon = self.rt.retrieve_recon(x_enc, x_mark_enc, batch_mask) # batch, seq_len, channel
                elif self.args.rag_strategy == 'whole':
                    x_recon = self.rt.retrieve_recon_whole(x_enc, x_mark_enc, batch_mask)
                x_enc = torch.where(batch_mask.unsqueeze(1).bool().to(x_enc.device), x_enc, x_recon)
            elif self.args.rag_type == 'no_rag':

                x_recon = x_enc   

        outputs = []
        for i in range(x_enc.shape[-1]):
            output = self.model.predict(x_enc[...,i].cpu().numpy())
            output = np.mean(output, axis=1)
            outputs.append(torch.Tensor(output).to(x_enc.device))
        dec_out = torch.stack(outputs, dim=-1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None