import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os
from rag.contr_itransformer import iTransformerContrastive

class missRetrieval():
    def __init__(self, configs, topk=10, temperature=0.1, device='cpu', retriever='iTransformer'):
        self.topk = topk
        self.temperature = temperature
        
        # 核心修改：初始化这两个变量存储在 CPU 内存中的数据
        self.kb_embed = None      # (N, C, D) 存放在 RAM
        self.kb_raw = None        # (N, L, C) 存放在 RAM
        
        self.n_train = 0
        self.retriever = retriever
        self.encoder = None
        self.device = device
        self.args = configs
        
    # 去掉保存到硬盘/读取的步骤。
    def training_encoder(self, args, train_loader, vali_loader):
        """保持不变，训练 Encoder"""

        # forecasting logic...
        # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(train_loader):
        args.dim_x_mark = 0 # 示例逻辑
        
        model = iTransformerContrastive(args, self.device) 
        params = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"Total parameters of encoder: {params / (1024 ** 2):.2f} MB")
        path = f"./pretrained_encoder/{args.task_name}/{args.data}/{args.retrieve_encoder}_{args.contrastive_loss}_{args.mask_ratio}_{model.d_model}_checkpoints.pt"

        if os.path.exists(path):
            print(f"Loading pretrained encoder from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Start training Our own {args.retrieve_encoder} Retrieval Encoder..., Contrastive Loss: {args.contrastive_loss}")
            model = model.fit(train_loader, vali_loader=vali_loader, path=path)
            # torch.save(model.state_dict(), path) 不再保存预训练Encoder。
            print(f"Training of Retrieval Encoder finished.")

        self.encoder = model

    def prepare_dataset(self, args, train_loader_unshuffled):
        """
        构建或加载 Knowledge Base，并将其完全驻留在 CPU 内存 (RAM) 中。
        """
        task_name = args.task_name
        dataset_name = args.data
        D = args.latent_dim
        N = len(train_loader_unshuffled.dataset) 
        C = args.enc_in 
        L = args.seq_len
        self.n_train = N

        print(f'Building Retrieval Knowledge Base (N={N}, C={C}, L={L}, D={D})...')

        # 3. 如果缓存不存在，在内存中创建并计算
        print("Constructing new KB in RAM...")
        
        # 直接在内存分配 (RAM)
        self.kb_embed = np.zeros((N, C+args.dim_x_mark, D), dtype=np.float32)
        self.kb_raw = np.zeros((N, L, C), dtype=np.float32)
        kb_embed_mem = self.kb_embed.nbytes / (1024 ** 2)
        kb_raw_mem = self.kb_raw.nbytes / (1024 ** 2)
        print(f"Memory - kb_embed: {kb_embed_mem:.2f} MB")
        print(f"Memory - kb_raw: {kb_raw_mem:.2f} MB")

        self.encoder.to(self.device)
        self.encoder.eval()

        offset = 0
        with torch.no_grad():
            # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_x_full) in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):
            for i, batch in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):
                if args.task_name == 'long_term_forecast':
                    batch_x_full = batch[6]
                    batch_x_mark = batch[3]
                elif args.task_name == 'imputation':
                    batch_x_full = batch[5]
                    batch_x_mark = batch[2]
                elif args.task_name == 'classification' or args.task_name == 'anomaly_detection':
                    batch_x_full = batch[1]
                    batch_x_mark = None
                x_full = batch_x_full.float().to(self.device)
                if batch_x_mark is not None:
                    x_mark = batch_x_mark.float().to(self.device)
                else:
                    x_mark = None
                
                B_current = x_full.shape[0]
                full_mask = torch.ones(B_current, C, device=self.device)

                # 存 Raw Data (CPU 操作)
                self.kb_raw[offset : offset + B_current] = x_full.cpu().numpy()

                # 生成 Embedding
                if self.retriever in ['Typology', 'iTransformer']: 
                    enc_output = self.encoder.get_representation(x_full, x_mark, mask=full_mask) 
                    self.kb_embed[offset : offset + B_current] = enc_output.cpu().numpy()
                else:
                    pass

                offset += B_current
        
        print("KB encoding finished.")

        # # 4. 保存到硬盘 (Persistence) 以备下次使用
        # # 这样下次运行就不用重新 Encode 了
        # print("Saving KB to disk for future usage...")
        
        # # 使用 memmap 进行写入，避免内存暴涨 (Dump RAM -> Disk)
        # fp_embed_save = np.memmap(embed_path, mode="w+", dtype=np.float32, shape=(N, C+args.dim_x_mark, D))
        # fp_embed_save[:] = self.kb_embed[:]
        # fp_embed_save.flush()
        
        # fp_raw_save = np.memmap(raw_path, mode="w+", dtype=np.float32, shape=(N, L, C))
        # fp_raw_save[:] = self.kb_raw[:]
        # fp_raw_save.flush()
        
        # print(f"KB saved to {cache_dir}")


    def retrieve_recon_whole(self, x, x_mark, mask):
        
        device = x.device
        B, L, C = x.shape
        D = self.args.latent_dim

        # ================= 1. 获取 Query Embedding =================
        with torch.no_grad():
            query_repr = self.encoder.get_representation(x, x_mark, mask=mask) # [B, C, D]

        # 【核心修改 A】：在循环外处理 Query
        # 1. Flatten: [B, C, D] -> [B, C*D]
        query_flat = query_repr.reshape(B, -1)
        
        # 2. Global Normalize: 对整个 C*D 维度做归一化
        # 这样计算出的才是 Global Cosine Similarity
        query_global_norm = F.normalize(query_flat, p=2, dim=1) 

        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)

         # 计算显存预算
        BYTES_PER_FLOAT = 4
        # 每次搬运 2GB 数据到 GPU (非常安全)
        MAX_GPU_BYTES = 2 * 1024**3 
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (C * D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving..."):
            end = min(start + kb_batch_size, N)

            # 加载 KB Chunk (CPU -> GPU)
            kb_chunk_cpu = torch.from_numpy(self.kb_embed[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) # [Chunk, C, D]

            # 【核心修改 B】：在循环内处理 KB Key
            # 1. Flatten Key: [Chunk, C, D] -> [Chunk, C*D]
            kb_flat = kb_chunk.reshape(kb_chunk.shape[0], -1)
            
            # 2. Global Normalize Key
            kb_global_norm = F.normalize(kb_flat, p=2, dim=1)

            # 3. 计算 Global Similarity (矩阵乘法)
            # [B, C*D] @ [Chunk, C*D].T -> [B, Chunk]
            sim = torch.matmul(query_global_norm, kb_global_norm.T)
            
            # 存入总表
            sim_final[:, start:end] = sim

        # 3. Top-K Selection
        topk_sim, topk_indices = torch.topk(sim_final, self.topk, dim=1)
        prob = F.softmax(topk_sim / self.temperature, dim=1) 

        # 4. Raw Data Retrieval (From RAM)
        # 将 GPU 上的索引转回 CPU
        indices_np = topk_indices.cpu().numpy() # (B, K)
        
        # [关键] 直接使用 Numpy Fancy Indexing 从 RAM 读取
        # self.kb_raw 是 (N, L, C) 的内存数组
        # 这一步比 memmap 快非常多
        retrieved_samples_np = self.kb_raw[indices_np] # Result: (B, K, L, C)
        
        # 转回 Tensor 并送入 GPU
        retrieved_samples = torch.from_numpy(retrieved_samples_np).to(device)

        # 5. Integration
        weights = prob.view(B, self.topk, 1, 1)
        x_recon = (weights * retrieved_samples).sum(dim=1)
        return x_recon


    def retrieve_recon(self, x, x_mark, mask):
        """
        Retrieval now happens fully in RAM (self.kb_embed) and GPU.
        """
        device = x.device
        B, L, C = x.shape
        D = self.args.latent_dim

        # 1. 获取 Query Embedding
        with torch.no_grad():
            query_repr = self.encoder.get_representation(x, x_mark, mask=mask)

        # (B, C, D) -> Normalize
        query_norm = F.normalize(query_repr, dim=-1)

        # 2. 检索逻辑 (Batch-wise to save GPU memory)
        # 虽然源数据在 RAM (快)，但为了防止 GPU OOM (计算 Cosine 时显存消耗大)，
        # 我们依然分块把 RAM 数据搬运到 GPU。
        
        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)
        
        # 计算显存预算
        BYTES_PER_FLOAT = 4
        # 每次搬运 2GB 数据到 GPU (非常安全)
        MAX_GPU_BYTES = 2 * 1024**3 
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (C * D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving and calculating sim...", leave=False):
            end = min(start + kb_batch_size, N)

            # [关键] 现在的切片操作是在 RAM 中进行的，极快！
            # self.kb_embed 是 numpy array, 不是 memmap
            kb_chunk_cpu = torch.from_numpy(self.kb_embed[start:end]) 
            
            # Move to GPU
            kb_chunk = kb_chunk_cpu.to(device) 
            kb_chunk_norm = F.normalize(kb_chunk, dim=-1) 

            # Compute Similarity: (B, C, D) x (Chunk, C, D) -> (B, Chunk, C)
            sim_per_channel = torch.einsum('bcd,ncd->bnc', query_norm, kb_chunk_norm)
            
            # Aggregate channels
            sim = sim_per_channel.mean(dim=2) # (B, Chunk)
            
            sim_final[:, start:end] = sim

        # 3. Top-K Selection
        topk_sim, topk_indices = torch.topk(sim_final, self.topk, dim=1)
        prob = F.softmax(topk_sim / self.temperature, dim=1) 

        # 4. Raw Data Retrieval (From RAM)
        # 将 GPU 上的索引转回 CPU
        indices_np = topk_indices.cpu().numpy() # (B, K)
        
        # [关键] 直接使用 Numpy Fancy Indexing 从 RAM 读取
        # self.kb_raw 是 (N, L, C) 的内存数组
        # 这一步比 memmap 快非常多
        retrieved_samples_np = self.kb_raw[indices_np] # Result: (B, K, L, C)
        
        # 转回 Tensor 并送入 GPU
        retrieved_samples = torch.from_numpy(retrieved_samples_np).to(device)

        # 5. Integration
        weights = prob.view(B, self.topk, 1, 1)
        x_recon = (weights * retrieved_samples).sum(dim=1)

        return x_recon