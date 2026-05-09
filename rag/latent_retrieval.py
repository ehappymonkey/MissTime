# 明天改成mean C的。
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os
import argparse # TimerXL 需要用到

class Retrieval():
    def __init__(self, configs, topk=10, temperature=0.1, device='cpu', retriever='TimerXL'):
        self.topk = topk
        self.temperature = temperature
        self.device = device
        self.args = configs
        self.retriever = retriever
        
        # 核心存储 (RAM)
        self.kb_embed = None      # (N, C, D) D=1024 for TimerXL
        self.kb_raw = None        # (N, L, C)
        self.n_train = 0
        
        self.encoder = None
        self.latent_dim = 0       # 自动获取维度

        # ================== 初始化 TimerXL Encoder ==================
        if self.retriever == 'TimerXL':
            print("Loading Pre-trained TimerXL as Encoder...")
            from models import timer_xl  # 假设 models 文件夹下有 timer_xl.py
            
            # 构造 TimerXL 需要的参数
            xl_args = argparse.Namespace()
            xl_args.input_token_len = 96      # 注意：确保这里和你的数据 seq_len 匹配
            xl_args.output_token_len = 96
            xl_args.d_model = 1024
            xl_args.n_heads = 8
            xl_args.e_layers = 8
            xl_args.d_ff = 2048
            xl_args.dropout = 0.1
            xl_args.activation = 'relu'
            xl_args.use_norm = True
            xl_args.flash_attention = False
            xl_args.covariate = False
            xl_args.output_attention = False
            
            # 初始化模型
            model = timer_xl.Model(xl_args)
            
            # 加载权重
            # ckpt_path = './checkpoint.pth' # 请确保路径正确，或者改为绝对路径
            ckpt_path = '/mnt/data/yuyuanfeng/projects/code125/rag/checkpoint.pth'
            if os.path.exists(ckpt_path):
                # map_location='cpu' 防止 GPU 显存不够，之后再 .to(device)
                state_dict = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"✅ TimerXL checkpoint loaded from {ckpt_path}")
            else:
                print(f"⚠️ Warning: Checkpoint {ckpt_path} not found! Using random init.")

            self.encoder = model
            self.encoder.to(self.device)
            self.encoder.eval() # 冻结模式
            
            # 设置 Embedding 维度
            self.latent_dim = xl_args.d_model # 1024
        else:
            raise NotImplementedError(f"Retriever {retriever} not implemented yet.")
        

    def prepare_dataset(self, args, train_loader_unshuffled):
        """
        构建 Knowledge Base (KB) 到内存 (RAM)。
        同时存储：
        1. kb_raw: 原始数据 (用于重构)
        2. kb_embed: TimerXL 提取的特征 (用于计算相似度)
        """
        N = len(train_loader_unshuffled.dataset) 
        C = args.enc_in 
        L = args.seq_len
        D = self.latent_dim # 1024
        self.n_train = N

        print(f'Building TimerXL Knowledge Base (Mean-Pooled) (N={N}, D={D})...')

        # 1. 在 RAM 分配空间
        # 大约大小估算: 50000 * 300 * 1024 * 4 bytes ≈ 60GB (对于 180G 内存完全没问题)
        self.kb_embed = np.zeros((N, D), dtype=np.float32) 
        self.kb_raw = np.zeros((N, L, C), dtype=np.float32)

        offset = 0
        self.encoder.eval()
        
        with torch.no_grad():
            # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_x_full) in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):
            for i, batch in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):

                if args.task_name == 'long_term_forecast':
                    batch_x_full = batch[6]
                elif args.task_name == 'imputation':
                    batch_x_full = batch[5]

                # batch_x_full: (B, L, C)
                batch_data = batch_x_full.float().to(self.device)
                B_current = batch_data.shape[0]

                # 1. 存 Raw Data (CPU)
                self.kb_raw[offset : offset + B_current] = batch_data.cpu().numpy()

                # 2. 计算 TimerXL Embedding
                # 输入: (B, L, C) -> 输出: (B, C, D_model)
                if self.retriever == 'TimerXL':
                    # 注意：TimerXL 可能对输入维度有要求，确保 batch_data 是 (B, L, C)
                    batch_repr = self.encoder.get_embedding(batch_data) 
                    batch_repr_mean = batch_repr.mean(dim=1)
                    
                    # 存入 RAM
                    self.kb_embed[offset : offset + B_current] = batch_repr_mean.cpu().numpy()

                offset += B_current
        
        print("✅ TimerXL Knowledge Base built and stored in RAM.")


    def retrieve_recon(self, x):
        """
        基于 TimerXL 特征检索，并重构序列。
        """
        device = x.device
        B, L, C = x.shape
        D = self.latent_dim

        # ================= 1. 获取 Query Embedding (TimerXL) =================
        self.encoder.eval()
        with torch.no_grad():
            # x: (B, L, C) -> query_repr: (B, C, D)
            # TimerXL 的 get_embedding 通常不需要 mask，它提取的是观测到的特征
            query_repr = self.encoder.get_embedding(x).mean(dim=1)


        query_norm = F.normalize(query_repr, p=2, dim=1) 

        # ================= 3. 检索循环 (Batch-wise Retrieval) =================
        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)

        # 显存预算控制
        BYTES_PER_FLOAT = 4
        MAX_GPU_BYTES = 2 * 1024**3 # 2GB
        # Chunk Size: 这里的维度是 C*D (例如 300*1024)
        kb_batch_size = max(1, int(MAX_GPU_BYTES // ( D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving (TimerXL)...", leave=False):
            end = min(start + kb_batch_size, N)

            # A. 加载 KB Embedding Chunk (RAM -> GPU)
            # self.kb_embed 是 (N, D)
            kb_chunk_cpu = torch.from_numpy(self.kb_embed[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) # [Chunk, D]

            
            kb_chunk_norm = F.normalize(kb_chunk, p=2, dim=1)

            # C. 计算 Similarity
            # [B, C*D] @ [Chunk, C*D].T -> [B, Chunk]
            sim = torch.matmul(query_norm, kb_chunk_norm.T)
            
            sim_final[:, start:end] = sim

        # ================= 4. Top-K & 重构 =================
        # Top-K
        topk_sim, topk_indices = torch.topk(sim_final, self.topk, dim=1)
        
        # Softmax
        prob = F.softmax(topk_sim / self.temperature, dim=1) 

        # Retrieve Raw Data
        indices_np = topk_indices.cpu().numpy() # (B, K)
        retrieved_samples_np = self.kb_raw[indices_np] # From RAM: (B, K, L, C)
        
        # Weighted Sum
        retrieved_samples = torch.from_numpy(retrieved_samples_np).to(device)
        weights = prob.view(B, self.topk, 1, 1)
        
        x_recon = (weights * retrieved_samples).sum(dim=1) # (B, L, C)
        
        return x_recon