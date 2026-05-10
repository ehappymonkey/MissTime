



import torch
import torch.nn.functional as F
import numpy as np
import copy
import math




class Retrieval():
    def __init__(self, topk=20, temperature=0.1):
        self.topk = topk
        self.temperature = temperature
        self.train_data = None      # (N, L, C)
        self.n_train = 0

    def prepare_dataset(self, train_data, task_name=None, dataset_name=None):
        print('Preparing Retrieval Dataset...')
        train_x_list = []
        for i in range(len(train_data)):
            if task_name == 'classification' or task_name == 'anomaly_detection':
                td = train_data[i][0]
            else:
                td = train_data[i][1]
            train_x_list.append(td)  # td: (seq_len, channels)
        
        self.train_data = torch.tensor(np.stack(train_x_list, axis=0)).float()  # (N, L, C)
        self.n_train = self.train_data.shape[0]
        print(f"KB built: {self.n_train} samples of shape {self.train_data.shape[1:]}")

    def retrieve_recon(self, x, observed_mask=None):
        """
         observed ， C_obs*L -> C_obs*L
        
        Args:
            x: (B, L, C) —— ，missing （ observed）
            observed_mask: (C,) —— 1  observed，0  missing
        
        Returns:
            x_recon: (B, L, C) —— （missing ，observed ）
        """
        device = x.device
        B, L, C = x.shape
        self.train_data = self.train_data.to(device)

    
        observed_mask = observed_mask.unsqueeze(0)  # (1, C)
        obs_idx = torch.where(observed_mask[0])[0]  # (C_obs,)

        x_obs = x[:, :, obs_idx]  # (B, L, C_obs)
        kb_obs = self.train_data[:, :, obs_idx]  # (N, L, C_obs)

        x_flat = x_obs.reshape(B, -1)  # (B, D)
        kb_flat = kb_obs.reshape(self.n_train, -1)  # (N, D)


        x_norm = x_flat - x_flat.mean(dim=1, keepdim=True)
        kb_norm = kb_flat - kb_flat.mean(dim=1, keepdim=True)


        x_norm = F.normalize(x_norm, dim=1)      # (B, D)
        kb_norm = F.normalize(kb_norm, dim=1)    # (N, D)
        sim = torch.mm(x_norm, kb_norm.t())      # (B, N)


        topk = min(self.topk, self.n_train)
        topk_sim, topk_indices = torch.topk(sim, topk, dim=1)  # (B, K)


        prob = F.softmax(topk_sim / self.temperature, dim=1)  # (B, K)
        kb_full = self.train_data.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, L, C)
        topk_samples = torch.gather(
            kb_full, 
            dim=1, 
            index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, C)
        )  # (B, K, L, C)


        x_recon = torch.sum(prob.unsqueeze(-1).unsqueeze(-1) * topk_samples, dim=1)  # (B, L, C)
        return x_recon.to(device)  # (B, L, C)


class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        

    def prepare_dataset(self, train_data):
        print('Preparing Retrieval Dataset...')
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            
            # if self.with_dec:
            #     y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            # else:
            #     y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        # self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        # self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]


    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # samples, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1) # (N, seq_len//g, C, g)  -> (N, seq_len//g, C)
            cur = cur.repeat_interleave(repeats=g, dim=1) # N, seq_len, C
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, N, L, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim


    def retrieve_recon(self, x, index, observed_mask=None, train=False):
        """
        x: (B, L, C) —— ，missing  0
        observed_mask: (C,) —— per-batch mask, 1=observed
        Returns: (B, L, C) —— final reconstruction (observed kept, missing imputed)
        """
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels


        observed_mask = observed_mask.expand(bsz, -1)  # (B, C)
        obs_idx = torch.where(observed_mask[0])[0]
        x_input = x[:, :, obs_idx]  # Batch, seq_len, C/Obs_C
        kb_input = self.train_data_all_mg[:, :, :, obs_idx] # G, Batch, seq_len, C/Obs_C
    

        x_mg, _ = self.decompose_mg(x_input) # G, B, seq_len, channel/obs_channel
        sim = self.periodic_batch_corr(kb_input.flatten(2), x_mg.flatten(2)) # G, B, seq_len*channel


        sim = sim.reshape(self.n_period * bsz, self.n_train)
        topm_index = torch.topk(sim, self.topm, dim=1).indices

        ranking_sim = torch.full_like(sim, float('-inf'))
        rows = torch.arange(sim.size(0), device=sim.device).unsqueeze(-1)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train)
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)  # (G, B, T)
        ranking_prob = ranking_prob.detach().cpu()


        x_data_all = self.train_data_all_mg.flatten(start_dim=2)  # (G, T, L*C)
        recon_all = torch.bmm(ranking_prob, x_data_all)
        recon_all = recon_all.reshape(self.n_period, bsz, -1, self.channels).to(x.device) # G, batch, seq_len, channel

        return recon_all


 