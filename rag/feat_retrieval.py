import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os

class Retrieval():
    def __init__(self, configs, topk=10, temperature=0.1, device='cpu', retriever='RawFeature'):
        self.topk = topk
        self.temperature = temperature
        
        # 修改：只需要 kb_raw，因为特征就是 raw data 本身
        self.kb_raw = None        # (N, L, C) 存放在 RAM
        
        self.n_train = 0
        self.retriever = retriever # 标记为 RawFeature
        self.device = device
        self.args = configs
        

    def prepare_dataset(self, args, train_loader_unshuffled):
        """
        构建 Knowledge Base (KB)。
        对于基于 Raw Feature 的检索，KB 就是训练集原始数据本身。
        """
        task_name = args.task_name
        dataset_name = args.data
        N = len(train_loader_unshuffled.dataset) 
        C = args.enc_in 
        L = args.seq_len
        self.n_train = N

        print(f'Building Raw Feature Knowledge Base (N={N}, C={C}, L={L})...')
        print("Loading raw data into RAM...")
        
        # 1. 在内存分配空间
        self.kb_raw = np.zeros((N, L, C), dtype=np.float32)

        offset = 0
        # 2. 遍历 DataLoader 填充数据
        with torch.no_grad():
            # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_x_full) in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):
            for i, batch in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):

                if args.task_name == 'long_term_forecast':
                    batch_x_full = batch[6]
                elif args.task_name == 'imputation':
                    batch_x_full = batch[4]
                    
                # 获取完整的 Ground Truth 作为 History
                x_full = batch_x_full.float() # 保持在 CPU 即可，最后一次性存入 numpy
                
                B_current = x_full.shape[0]

                # 存 Raw Data (CPU 操作)
                self.kb_raw[offset : offset + B_current] = x_full.numpy()

                offset += B_current
        
        print("Raw Knowledge Base built.")


    def retrieve_recon(self, x):
        """
        基于输入 x 的原始特征与 KB 计算相似度，并检索重构。
        """
        device = x.device
        B, L, C = x.shape
        
        # ================= 1. 准备 Query (Raw Features) =================
        # x: [B, L, C]
        
        query_data = x

        # 1. Flatten: [B, L, C] -> [B, L*C]
        # 将整个序列打平作为一个特征向量
        query_flat = query_data.reshape(B, -1)
        
        # 2. Global Normalize: 归一化以便计算 Cosine Similarity
        query_global_norm = F.normalize(query_flat, p=2, dim=1) 

        # ================= 2. 检索循环 (Batch-wise Retrieval) =================
        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)

        # 显存预算控制
        BYTES_PER_FLOAT = 4
        # 每次搬运 2GB 数据到 GPU
        MAX_GPU_BYTES = 2 * 1024**3 
        # kb_chunk 大小：因为是 Raw Data，维度是 L*C
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (L * C * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving (Raw Feature)...", leave=False):
            end = min(start + kb_batch_size, N)

            # A. 加载 KB Chunk (CPU RAM -> GPU)
            # self.kb_raw 是 [N, L, C]
            kb_chunk_cpu = torch.from_numpy(self.kb_raw[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) # [Chunk, L, C]

            # B. 处理 Key (Flatten & Normalize)
            # [Chunk, L, C] -> [Chunk, L*C]
            kb_flat = kb_chunk.reshape(kb_chunk.shape[0], -1)
            
            # Global Normalize Key
            kb_global_norm = F.normalize(kb_flat, p=2, dim=1)

            # C. 计算 Global Similarity (矩阵乘法)
            # Query [B, D] @ Key_T [D, Chunk] -> [B, Chunk] (D = L*C)
            sim = torch.matmul(query_global_norm, kb_global_norm.T)
            
            # 存入总表
            sim_final[:, start:end] = sim

        # ================= 3. Top-K Selection =================
        # topk_indices: (B, K)
        topk_sim, topk_indices = torch.topk(sim_final, self.topk, dim=1)
        
        # Softmax 计算权重
        prob = F.softmax(topk_sim / self.temperature, dim=1) 

        # ================= 4. 获取原始数据 (Retrieve) =================
        # 将 GPU 上的索引转回 CPU
        indices_np = topk_indices.cpu().numpy() # (B, K)
        
        # 从 RAM 读取 Top-K 样本 (Fancy Indexing)
        retrieved_samples_np = self.kb_raw[indices_np] # Result: (B, K, L, C)
        
        # 转回 Tensor 并送入 GPU
        retrieved_samples = torch.from_numpy(retrieved_samples_np).to(device)

        # ================= 5. 加权融合 (Integration) =================
        # weights: (B, K, 1, 1)
        weights = prob.view(B, self.topk, 1, 1)
        
        # 加权求和得到 x_recon
        x_recon = (weights * retrieved_samples).sum(dim=1) # (B, L, C)
        
        return x_recon