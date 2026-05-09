# 1. 离线检索改为在线检索。
# 2. 检索用于更新（填补）input feature而不是结果。
# 3. 用缺失representations的embedding拼接检索历史缺失representations, 填补获得完整的representations。使用Moment作为encoder。


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from momentfm import MOMENTPipeline
from tqdm import tqdm 
import os



class latentRetrieval():
    def __init__(self, topk=20, temperature=0.1, device='cpu'):
        self.topk = topk
        self.temperature = temperature
        self.train_data_full = None      # (N, L, C)
        self.kb_repr = None              # (N, C, P, D)
        self.n_train = 0

        encoder = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode to learn representations
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )
        encoder.init()
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
        ).to(self.device)  # (N, L, C)

        self.encoder.to(self.device)
        N, L, C = self.train_data_full.shape
        self.n_train = N
        print(f"Raw KB built: {N} samples of shape {(L, C)}")

        # 探测 P, D (C_enc 不再需要)
        print("🔍 Probing model output shape with a single sample...")
        with torch.no_grad():
            sample = self.train_data_full[:1].permute(0, 2, 1)
            sample_out = self.encoder(x_enc=sample.to(self.device), reduction="none")
            _, C_enc, P, D = sample_out.embeddings.shape
        print(f"Detected: P={P}, D={D} (C={C_enc} will be pooled)")

        self.P = P
        self.D = D

        # 缓存路径 (存储 (N, P, D) 而非 (N, C, P, D))
        cache_dir = f"./latentKBCache/{task_name}/{dataset_name}"
        os.makedirs(cache_dir, exist_ok=True)
        KB_cache_path = os.path.join(cache_dir, f"repr_KB_pooled_N{N}_P{P}_D{D}.dat")

        # 加载缓存
        if os.path.exists(KB_cache_path):
            print(f"Loading cached POOLED latent KB from {KB_cache_path}")
            self.kb_repr_mm = np.memmap(
                KB_cache_path,
                mode="r",
                dtype=np.float32,
                shape=(N, P, D)  # 👈 关键：移除 C 维度
            )
            print(f"Loaded pooled KB: shape={self.kb_repr_mm.shape}")
            return

        # 编码并存储 POOLED 表示
        print("Encoding and pooling training data with MOMENT...")
        
        # 创建 (N, P, D) memmap
        kb_memmap = np.memmap(
            KB_cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(N, P, D)  # 👈 关键：移除 C
        )

        offset = 0
        with torch.no_grad():
            for start_idx in tqdm(range(0, N, batch_size), desc="Encoding batches"):
                end_idx = min(start_idx + batch_size, N)
                
                # 编码整个 batch
                batch_data = self.train_data_full[start_idx:end_idx]
                batch_enc = batch_data.permute(0, 2, 1).to(self.device)
                enc_output = self.encoder(x_enc=batch_enc, reduction="none")
                
                # 👇 关键：在 GPU 上直接池化
                batch_pooled = enc_output.embeddings.mean(dim=1)  # (B, P, D)
                batch_pooled = F.normalize(batch_pooled, dim=-1)  # 👈 预归一化 (可选但推荐)
                
                # 写入磁盘
                kb_memmap[offset:offset + batch_pooled.shape[0]] = batch_pooled.cpu().numpy()
                offset += batch_pooled.shape[0]
                
                del batch_enc, enc_output, batch_pooled
                torch.cuda.empty_cache()

        # 将 memmap 挂到对象上（只读视图，不占 RAM）
        self.kb_repr_mm = np.memmap(
            KB_cache_path,
            mode="r",
            dtype=np.float32,
            shape=(N, C_enc, P, D)
        )

    #     print(f"Latent KB saved safely to {KB_cache_path}")
   
    # def prepare_dataset(self, train_data, task_name=None, dataset_name=None, batch_size=4):
    #     print('Preparing Retrieval Dataset...')

    #     # --------------------------------------------------
    #     # 1. 构建原始训练数据 (与原逻辑一致)
    #     # --------------------------------------------------
    #     train_x_list = []
    #     for i in range(len(train_data)):
    #         if task_name in ['classification', 'anomaly_detection']:
    #             td = train_data[i][0]
    #         else:
    #             td = train_data[i][1]
    #         train_x_list.append(td)  # (L, C)

    #     self.train_data_full = torch.tensor(
    #         np.stack(train_x_list, axis=0),
    #         dtype=torch.float32
    #     )  # (N, L, C)

    #     self.encoder.to(self.device)

    #     N, L, C = self.train_data_full.shape
    #     self.n_train = N
    #     print(f"Raw KB built: {N} samples of shape {(L, C)}")

    #     print("🔍 Probing model output shape with a single sample...")
    #     self.encoder.to(self.device)
    #     self.train_data_full = self.train_data_full.to(self.device)
        
    #     # 4.1 先跑一个 batch，确定 (C, P, D)
    #     with torch.no_grad():
    #         sample = self.train_data_full[:1].permute(0, 2, 1)
    #         sample_out = self.encoder(x_enc=sample, reduction="none")
    #         _, C_enc, P, D = sample_out.embeddings.shape

    #     print(f"Latent repr shape per sample: (C={C_enc}, P={P}, D={D})")

    #     # --------------------------------------------------
    #     # 2. 缓存路径
    #     # --------------------------------------------------
    #     cache_dir = f"./latentKBCache/{task_name}/{dataset_name}"
    #     os.makedirs(cache_dir, exist_ok=True)

    #     # 使用 .dat 作为 memmap 文件
    #     KB_cache_path = os.path.join(cache_dir, f"repr_KB_{N}.dat")
    #     # meta_path = os.path.join(cache_dir, f"repr_KB_{N}_meta.pt")

    #     # --------------------------------------------------
    #     # 3. 若缓存已存在：直接加载（O(1) RAM）
    #     # --------------------------------------------------
    #     if os.path.exists(KB_cache_path): # and os.path.exists(meta_path):
    #         print(f"Loading cached latent KB from {KB_cache_path}")

    #         # meta = torch.load(meta_path)
    #         # shape = meta["shape"]
    #         # dtype = meta["dtype"]

    #         self.kb_repr_mm = np.memmap(
    #             KB_cache_path,
    #             mode="r",
    #             dtype=np.float32,
    #             shape=(self.n_train, C_enc, P, D)
    #         )

    #         print(f"Latent KB loaded (memmap): shape={(N, C_enc, P, D)}")
    #         return

    #     # --------------------------------------------------
    #     # 4. 编码并“边算边写磁盘”（关键改动）
    #     # --------------------------------------------------
    #     print("Encoding training data with MOMENT...")

    
    #     # 4.2 创建 memmap（一次性，不占 RAM）
    #     kb_memmap = np.memmap(
    #         KB_cache_path,
    #         mode="w+",
    #         dtype=np.float32,
    #         shape=(N, C_enc, P, D)
    #     )

    #     offset = 0
    #     with torch.no_grad():
    #         for start_idx in tqdm(range(0, N, batch_size), desc="Calculating similarity through batches..."):
    #             end_idx = min(start_idx + batch_size, N)

    #             batch_data = self.train_data_full[start_idx:end_idx]  # (B, L, C)
    #             batch_enc = batch_data.permute(0, 2, 1)               # (B, C, L)

    #             enc_output = self.encoder(x_enc=batch_enc, reduction="none")
    #             batch_repr = enc_output.embeddings.cpu().numpy()      # (B, C, P, D) # 是不是存入mean channels后的B, P, D就可以了？

    #             # 🔴 关键：直接写入磁盘
    #             kb_memmap[offset:offset + batch_repr.shape[0]] = batch_repr
    #             offset += batch_repr.shape[0]

    #             del batch_enc, enc_output, batch_repr
    #             torch.cuda.empty_cache()


    #     # 将 memmap 挂到对象上（只读视图，不占 RAM）
    #     self.kb_repr_mm = np.memmap(
    #         KB_cache_path,
    #         mode="r",
    #         dtype=np.float32,
    #         shape=(N, C_enc, P, D)
    #     )

    #     print(f"Latent KB saved safely to {KB_cache_path}")

    def retrieve_recon(self, x, observed_mask=None):
        device = x.device
        B, L, C = x.shape
        
        # 1. 编码查询样本 (保留 C_obs 维度)
        obs_idx = torch.where(observed_mask)[0]
        x_obs = x[:, :, obs_idx]
        x_query = x_obs.permute(0, 2, 1).to(self.device)  # 👈 确保在 GPU
        
        with torch.no_grad():
            output = self.encoder(x_enc=x_query, reduction='none')
            query_repr = output.embeddings  # (B, C_obs, P, D)
        
        # 2. 池化 + 归一化查询表示
        query_pooled = query_repr.mean(dim=1)  # (B, P, D)
        query_norm = F.normalize(query_pooled, dim=-1)  # (B, P, D)


        if hasattr(self, 'kb_on_gpu') and self.kb_on_gpu:
            # GPU 检索 (超快!)
            sim_final = torch.einsum('bpd, npd -> bnp', query_norm, self.kb_repr_gpu)
            sim_final = sim_final.mean(dim=2)
        else:
            # 3. 分批次检索 
            N = self.n_train
            P, D = query_norm.shape[1], query_norm.shape[2]
            
            # 动态计算 batch_size (现在更小!)
            MAX_GPU_BYTES = 40 * 1024**3
            kb_batch_size = max(1, int(MAX_GPU_BYTES // (P * D * 4)))
            # 👆 注意：不再需要 C_obs! 因为 KB 已经是 (N, P, D)

            # 预分配相似度矩阵
            sim_final = torch.zeros(B, N, device=device)
            
            # 分块检索
            # for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving from KB..."):
            for start in range(0, N, kb_batch_size):
                end = min(start + kb_batch_size, N)
                
                # 直接加载预计算的 (P, D) 表示
                kb_batch = torch.from_numpy(
                    self.kb_repr_mm[start:end]  # 👈 (batch_size, P, D)
                ).to(device)
                
                # 点积 = 余弦相似度 (因为已预归一化)
                sim = torch.einsum('bpd, npd -> bnp', query_norm, kb_batch)  # (B, batch_size)
                sim_final[:, start:end] = sim

        # 4. Top-K 检索 (后续不变)
        topk = min(self.topk, self.n_train)
        topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)
        prob = F.softmax(topk_sim / self.temperature, dim=1)
        
        retrieved_samples = self.train_data_full[topk_indices]
        weights = prob.view(B, topk, 1, 1)
        x_recon = (weights * retrieved_samples).sum(dim=1)
        
        return x_recon.to(device)
        
    # def retrieve_recon(self, x, observed_mask=None):
    #     """
    #     根据 observed 变量的representation (P, D) 检索KB中最相似样本(P, D)，并重构完整序列。
        
    #     Args:
    #         x: (B, L, C) —— 输入，missing 处可为任意值（因只用 observed）
    #         observed_mask: (C,) —— 1 表示 observed，0 表示 missing
        
    #     Returns:
    #         x_recon: (B, L, C) —— 重构结果（missing 变量被填充，observed 保留原值）
    #     """

    #     device = x.device
    #     B, L, C = x.shape
        
    #     obs_idx = torch.where(observed_mask)[0]  # (C_obs,)
    #     x_obs = x[:, :, obs_idx]  # (B, L, C_obs)
    #     x_query = x_obs.permute(0, 2, 1)  # (B, C_obs, L)

    #     with torch.no_grad():
    #         x_enc = x_query # .permute(0, 2, 1).to(self.device)  # (B, C_obs, L)
    #         output = self.encoder(
    #             x_enc=x_enc,
    #             reduction='none'
    #         )
    #         query_repr = output.embeddings  # (B, C_obs, P, D)
    #         C_obs, P, D = query_repr.shape[1], query_repr.shape[2], query_repr.shape[3]

        
    #     query_pooled = query_repr.mean(dim=1)  # (B, P, D) 
    #     query_norm = F.normalize(query_pooled, dim=-1)  # (B, P, D)

    #     BYTES_PER_FLOAT = 4
    #     MAX_GPU_BYTES = 40*1024**3  # 40 GB
    #     kb_batch_size = max(
    #         1,
    #         int(MAX_GPU_BYTES // (C_obs * P * D * BYTES_PER_FLOAT))
    #     )
        

    #     N = self.n_train
    #     sim_chunks = []   # 用来存每个 batch 的 (B, max_batch)
    #     for start in tqdm(range(0, N, kb_batch_size), desc="Quering Similarity through KB batches..."):
    #         end = min(start + kb_batch_size, N)

    #         kb_batch = torch.from_numpy(
    #             self.kb_repr_mm[start:end][:, obs_idx, :, :]
    #         ).to(device)       

    #         kb_pooled = kb_batch.mean(dim=1)        # (max_batch, P, D) 
    #         kb_norm = F.normalize(kb_pooled, dim=-1)        # (max_batch, P, D)
        
    #         # 计算每 patch 的相似度 (B, max_batch, P)
    #         sim = torch.einsum('bpd,npd->bnp', query_norm, kb_norm)
    #         sim = sim.mean(dim=2)  # (B, max_batch) - 在max_batch内的sim
    #         sim_chunks.append(sim)

    #     # 6. Top-K 检索
    #     topk = min(self.topk, self.n_train)
    #     sim_final = torch.cat(sim_chunks, dim=1)  # (B, N)
    #     topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)  # (B, K)
    #     prob = F.softmax(topk_sim / self.temperature, dim=1)  # (B, K)

    #     # 7. 重构完整序列
    #     retrieved_samples = self.train_data_full[topk_indices]  # (B, K, L, C)
    #     weights = prob.view(B, topk, 1, 1)  # (B, K, 1, 1)
    #     x_recon = (weights * retrieved_samples).sum(dim=1)  # (B, L, C)

        
    #     return x_recon.to(device)

