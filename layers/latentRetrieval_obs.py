import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from momentfm import MOMENTPipeline
from tqdm import tqdm 
import os


class latentRetrieval():
    def __init__(self, topk=20, temperature=0.1, device='cpu', retriever='TimerXL'):
        self.topk = topk
        self.temperature = temperature
        self.train_data_full = None      # (N, L, C)
        self.kb_repr = None              # (N, C, P, D)
        self.n_train = 0

        self.retriever = retriever
        if retriever == 'Moment':
            # Moment 作为检索器
            encoder = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode to learn representations
            # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
            )
            encoder.init()
        elif retriever == 'TimerXL':
            import argparse
            from models import timer_xl
            args = argparse.Namespace()
            args.input_token_len = 96
            args.output_token_len = 96
            args.d_model = 1024
            args.n_heads = 8
            args.e_layers = 8
            args.d_ff = 2048
            args.dropout = 0.1
            args.activation = 'relu'
            args.use_norm = True
            args.flash_attention = False
            args.covariate = False
            args.output_attention = False
            model = timer_xl.Model(args)
            # download the checkpoint from https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/
            model.load_state_dict(torch.load('checkpoint.pth'))
            encoder = model

        self.encoder = encoder
        self.device = device
        
    def load_kb_to_gpu(self):
        """将 KB 从磁盘加载到 GPU (仅当足够小时)"""
        kb_size_gb = self.n_train * self.P * self.D * 4 / 1e9
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if kb_size_gb < gpu_mem_gb * 0.8:
            print(f"🚀 Loading KB to GPU ({kb_size_gb:.2f} GB / {gpu_mem_gb:.1f} GB)")
            self.kb_repr_gpu = torch.from_numpy(
                np.array(self.kb_repr_mm)  # 加载全量到 CPU
            ).to(self.device)  # 移到 GPU
            self.kb_on_gpu = True
            print("✅ KB loaded to GPU")
        else:
            print(f"⚠️ KB too large ({kb_size_gb:.2f} GB) for GPU, keeping on disk")
            self.kb_on_gpu = False

    
    def prepare_dataset(self, train_data, task_name=None, dataset_name=None, batch_size=4):
        print('Preparing Retrieval Dataset...')

        # --------------------------------------------------
        # 1. 构建原始训练数据 (与原逻辑一致)
        # --------------------------------------------------
        train_x_list = []
        for i in range(len(train_data)):
            if task_name in ['classification', 'anomaly_detection']:
                td = train_data[i][0]
            else:
                td = train_data[i][1]
            train_x_list.append(td)  # (L, C)

        self.train_data_full = torch.tensor(
            np.stack(train_x_list, axis=0),
            dtype=torch.float32
        )  # (N, L, C)

        self.encoder.to(self.device)

        N, L, C = self.train_data_full.shape
        self.n_train = N
        print(f"Raw KB built: {N} samples of shape {(L, C)}")


        if self.retriever == 'Moment':
            print("🔍 Probing model output shape with a single sample...")
            self.encoder.to(self.device)
            self.train_data_full = self.train_data_full.to(self.device)
            
            # 跑一个 batch，确定 (C, P, D)
            with torch.no_grad():
                sample = self.train_data_full[:1].permute(0, 2, 1)
                sample_out = self.encoder(x_enc=sample, reduction="none")
                _, C, P, D = sample_out.embeddings.shape

            print(f"Latent repr shape per sample: (C={C}, P={P}, D={D})")
        
        elif self.retriever == 'TimerXL':
            print("🔍 Probing model output shape with a single sample...")
            self.encoder.to(self.device)
            self.train_data_full = self.train_data_full.to(self.device)
            
            # 跑一个 batch，确定 (C, D)
            with torch.no_grad():
                sample = self.train_data_full[:1]  # (1, L, C)
                sample_out = self.encoder.get_embedding(sample)  # (1, C, D)
                _, C, D = sample_out.shape
                P = 1  # Timer-XL 每变量只有一个表示

            print(f"Latent repr shape per sample: (C={C}, P={P}, D={D})")


        cache_dir = f"./latentKB_{self.retriever}/{task_name}/{dataset_name}"
        os.makedirs(cache_dir, exist_ok=True)
        KB_cache_path = os.path.join(cache_dir, f"repr_KB_{N}_last_Patch.dat")
    

        if os.path.exists(KB_cache_path): 
            print(f"Loading cached latent KB from {KB_cache_path}")
            self.kb_repr_mm = np.memmap(
                KB_cache_path,
                mode="r",
                dtype=np.float32,
                shape=(self.n_train, C, D)
            )
            print(f"Latent KB loaded (memmap): shape={(N, C, D)}")
            return

        # 创建latentKB
        print(f"Encoding training data to latentKB with {self.retriever}...")
        kb_memmap = np.memmap(
            KB_cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(N, C, D)
        )

        offset = 0
        with torch.no_grad():
            for start_idx in tqdm(range(0, N, batch_size), desc="Encoding representations through batches..."):
                end_idx = min(start_idx + batch_size, N)

                batch_data = self.train_data_full[start_idx:end_idx]  # (B, L, C)

                if self.retriever == 'Moment':
                    batch_enc = batch_data.permute(0, 2, 1)               # (B, C, L)
                    enc_output = self.encoder(x_enc=batch_enc, reduction="none")
                    batch_repr = enc_output.embeddings[:, :, -1, :].cpu().numpy()      # (B, C, P->1, D) 
                    # 切换为Mean Patch看看效果。
                elif self.retriever == 'TimerXL':
                    batch_enc = batch_data.to(self.device)  # (B, L, C)
                    batch_repr = self.encoder.get_embedding(batch_enc)  # (B, C, D_model)
                    batch_repr = batch_repr.cpu().numpy()  # (B, C, D_model)
                kb_memmap[offset:offset + batch_repr.shape[0]] = batch_repr
                offset += batch_repr.shape[0]

                del batch_enc, batch_repr
                torch.cuda.empty_cache()


        # 将 memmap 挂到对象上（只读视图，不占 RAM）
        self.kb_repr_mm = np.memmap(
            KB_cache_path,
            mode="r",
            dtype=np.float32,
            shape=(N, C, D)
        )

        print(f"Latent KB saved safely to {KB_cache_path}")

    def retrieve_recon(self, x, observed_mask=None):
            """
            根据 observed 变量的representation (P, D) 检索KB中最相似样本(P, D)，并重构完整序列。
            
            Args:
                x: (B, L, C) —— 输入，missing 处可为任意值（因只用 observed）
                observed_mask: (C,) —— 1 表示 observed，0 表示 missing
            
            Returns:
                x_recon: (B, L, C) —— 重构结果（missing 变量被填充，observed 保留原值）
            """

            device = x.device
            B, L, C = x.shape
            
            obs_idx = torch.where(observed_mask)[0]  # (C_obs,)
            x_obs = x[:, :, obs_idx]  # (B, L, C_obs)

            if self.retriever == 'Moment':
                x_query = x_obs.permute(0, 2, 1)  # (B, C_obs, L)
                with torch.no_grad():
                    x_enc = x_query # .permute(0, 2, 1).to(self.device)  # (B, C_obs, L)
                    output = self.encoder(
                        x_enc=x_enc,
                        reduction='none'
                    )
                    query_repr = output.embeddings  # (B, C_obs, P, D)
                    C_obs, P, D = query_repr.shape[1], query_repr.shape[2], query_repr.shape[3]
                query_repr = query_repr[:, :, -1, :]  # 只用最后一个 Patch 的表示 (B, C, 1, D)
                # 改成mean Patch看看效果
                # query_pooled = query_repr.mean(dim=1)  # (B, P, D) 
                # query_norm = F.normalize(query_repr, dim=-1)  # (B, P, D)
                query_flat = query_repr.reshape(B, -1)  # (B, C_obs*D)

            elif self.retriever == 'TimerXL':
                x_query = x_obs  # (B, L, C_obs)
                with torch.no_grad():
                    x_enc = x_query.to(self.device)  # (B, L, C_obs)
                    query_repr = self.encoder.get_embedding(x_enc)  # (B, C_obs, D_model)
                    C_obs, D = query_repr.shape[1], query_repr.shape[2]
                query_flat = query_repr.reshape(B, -1)  # (B, C_obs*D)

            # 准备好了query_repr B,C_obs,D和 query_flat B,C_obs*D两种表示。
            BYTES_PER_FLOAT = 4
            MAX_GPU_BYTES = 5*1024**3  # 40 GB
            kb_batch_size = max(
                1,
                int(MAX_GPU_BYTES // (C * 1 * D * BYTES_PER_FLOAT))
            )
            
            N = self.n_train
            sim_final = torch.zeros(B, N, device=device)
            obs_idx = obs_idx.cpu().numpy().tolist()
            for start in tqdm(range(0, N, kb_batch_size)):
                end = min(start + kb_batch_size, N)

                kb_batch = torch.from_numpy(
                    self.kb_repr_mm[start:end][:, obs_idx, :]
                ).to(device)        # (max_batch, C_obs, D) 

                # # 基于C_obs*D整体距离。
                # kb_flat = kb_batch.reshape(kb_batch.shape[0], -1) # (max_batch, C_obs*D)
                # dists = torch.cdist(query_flat, kb_flat, p=2)
                # sim_final[:, start:end] = -dists  

                # 基于每个C_obs coine相似度求平均。
                kb_norm = F.normalize(kb_batch, dim=-1)        # (max_batch, C_obs, D)
                query_norm = F.normalize(query_repr, dim=-1)        # (B, C_obs, D)
                sim = torch.einsum('bcd,ncd->bnc', query_norm, kb_norm)  # (B, max_batch, C_obs)
                sim = sim.mean(dim=2)  # (B, max_batch) - 在max_batch内的sim
                sim_final[:, start:end] = sim

                # # # 基于每个C_obs 距离求平均。
                # query_perm = query_repr.permute(1, 0, 2) 
                # kb_perm = kb_batch.permute(1, 0, 2)
                # dists_per_channel = torch.cdist(query_perm, kb_perm, p=2)
                # dists_bnc = dists_per_channel.permute(1, 2, 0)
                # mean_dist = dists_bnc.mean(dim=2)
                # sim_final[:, start:end] = -mean_dist
                
            # 6. Top-K 检索
            topk = self.topk
            topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)  # (B, K)
            prob = F.softmax(topk_sim / self.temperature, dim=1)  # (B, K)

            # 7. 重构完整序列
            retrieved_samples = self.train_data_full[topk_indices]  # (B, K, L, C)
            weights = prob.view(B, topk, 1, 1)  # (B, K, 1, 1)
            x_recon = (weights * retrieved_samples).sum(dim=1)  # (B, L, C)

            
            return x_recon.to(device)


