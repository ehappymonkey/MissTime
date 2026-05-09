# 支持iTransformer系列自己训练的Retrieve_Encoder。

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os
from rag.contr_itransformer import iTransformerContrastive

class missRetrieval():
    def __init__(self, configs,  topk=20, temperature=0.1, device='cpu', retriever='iTransformer'):
        self.topk = topk
        self.temperature = temperature
        self.train_data_full = None      # (N, L, C)
        self.kb_repr = None              # (N, C, P, D)
        self.n_train = 0

        self.retriever = retriever

        self.encoder = None
        self.device = device
        self.args = configs
        
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


    def training_encoder(self, args, train_loader, vali_loader):
 
        # 一个任务和数据集下只有一个Encoder就可以了。
        # 不同missing ratios下要有不同的Encoder！

        # forecasting
        for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(train_loader):
            # dim_x_mark = batch_x_mark.shape[2]
            # args.dim_x_mark = dim_x_mark
            args.dim_x_mark = 0
        model = iTransformerContrastive(args, self.device) 
        path = f"./pretrained_encoder/{args.task_name}/{args.data}/{args.retrieve_encoder}_{args.contrastive_loss}_{args.mask_ratio}_{model.d_model}_checkpoints.pt"

        if os.path.exists(path):
            print(f"Loading pretrained encoder from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print("Start training Our own Retrieval Encoder...")
            model = model.fit(train_loader, vali_loader=vali_loader, path=path)
            torch.save(model.state_dict(), path)
            print(f"Training of Retrieval Encoder finished and saved to {path}.")

        self.encoder = model

    def prepare_dataset(self, args, train_loader_unshuffled):
        """
        train_loader_unshuffled: 必须是一个 shuffle=False 的 DataLoader！
        用于按顺序提取训练集的所有数据构建 Knowledge Base。
        """
        task_name = args.task_name
        dataset_name = args.data
        D = args.latent_dim
        N = len(train_loader_unshuffled.dataset) # 获取样本总数
        C = args.enc_in # 变量数量
        L = args.seq_len
        self.n_train = N

        print(f'Building Retrieval Knowledge Base (N={N}, C={C}, L={L}, D={D})...')

        # 1. 定义缓存路径
        cache_dir = f"./latentKB/{task_name}/{dataset_name}"
        os.makedirs(cache_dir, exist_ok=True)
        
        # 我们需要存两个文件：
        # 1. Embedding: 用于检索 (N, C, D) 或者是 (N, C*D) 取决于你是否Flatten，这里假设保持结构
        embed_path = os.path.join(cache_dir, f"kb_embed_{args.retrieve_encoder}_{args.mask_ratio}_{D}.dat")
        # 2. Raw Data: 用于填补 (N, L, C)
        raw_path = os.path.join(cache_dir, f"kb_raw_data_{args.retrieve_encoder}_{args.mask_ratio}_{D}.dat")

        # 2. 检查缓存是否存在
        if os.path.exists(embed_path) and os.path.exists(raw_path):
            print(f"Loading cached KB from {cache_dir}...")
            # 挂载 Memmap (Read-only), 使用iTransformer的Encoder变量维度都等于C+dim_x_mark。
            self.kb_embed_mm = np.memmap(embed_path, mode="r", dtype=np.float32, shape=(N, C+args.dim_x_mark, D))
            self.kb_raw_mm = np.memmap(raw_path, mode="r", dtype=np.float32, shape=(N, L, C))
            return

        # 3. 创建 Memmap 文件 (Write mode)
        print("Constructing new KB...")
        fp_embed = np.memmap(embed_path, mode="w+", dtype=np.float32, shape=(N, C+args.dim_x_mark, D))
        fp_raw = np.memmap(raw_path, mode="w+", dtype=np.float32, shape=(N, L, C))

        self.encoder.to(self.device)
        self.encoder.eval() # 必须 Eval 模式

        offset = 0
        with torch.no_grad():
            # 遍历 DataLoader (不打乱的!)
            # 注意解包逻辑，根据你的 Dataset 具体的 return 内容调整
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_x_full) in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):
                
                # 这里的 batch_x_full 是完整的 Ground Truth
                # 我们用它来构建 KB，因为它代表了"Ideal History"
                
                # 1. 准备数据
                # x_full: [B, L, C]
                x_full = batch_x_full.float().to(self.device)
                
                # x_mark: [B, L, D_mark]
                if batch_x_mark is not None:
                    x_mark = batch_x_mark.float().to(self.device)
                
                # 2. 构造 Full Mask (全是 1)
                # KB 里的数据是完整的，所以 mask 全为 1
                # 形状 [B, C]
                B_current = x_full.shape[0]
                full_mask = torch.ones(B_current, C, device=self.device)

                # 3. 存 Raw Data
                batch_raw_np = x_full.cpu().numpy()
                fp_raw[offset : offset + B_current] = batch_raw_np

                # 4. 生成 Embedding
                if self.retriever == 'Typology' or self.retriever == 'iTransformer': # 你的 Topology-Aware Encoder
                    # get_representation 内部会处理 mask embedding 和 variate embedding
                    # 注意：对于 Full View，我们传入全 1 的 mask
                    enc_output = self.encoder.get_representation(x_full, x_mark, mask=full_mask) 
                    # enc_output shape 应该是 [B, C, D] (未Flatten) 或 [B, C*D] (Flatten)
                    
                    batch_embed_np = enc_output.cpu().numpy()
                    fp_embed[offset : offset + B_current] = batch_embed_np
                
                else:
                    # 其他 Encoder 的逻辑
                    pass

                offset += B_current
        
        # 4. 刷新硬盘并重新挂载为只读
        fp_embed.flush()
        fp_raw.flush()
        
        self.kb_embed_mm = np.memmap(embed_path, mode="r", dtype=np.float32, shape=(N, C+args.dim_x_mark, D))
        self.kb_raw_mm = np.memmap(raw_path, mode="r", dtype=np.float32, shape=(N, L, C))
        
        print("KB construction finished.")
        
    def prepare_dataset_old(self, args, train_data, batch_size=4):
        task_name = args.task_name
        dataset_name = args.data

        print('Preparing Retrieval Dataset...')

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
        D = args.latent_dim

        
        # 不同任务/数据集，不同missing ratios下面也要有不同的KB！
        # 相同数据集N, C是确定的，D和mask_ratio不确定。
        # 因此路径和pretrain_encoder保持一致即可。
        cache_dir = f"./latentKB/{task_name}/{dataset_name}"
        os.makedirs(cache_dir, exist_ok=True)
        KB_cache_path = os.path.join(cache_dir, f"dataset_name_{args.retrieve_encoder}_{args.contrastive_loss}_{args.mask_ratio}_{D}.dat")
    

        if os.path.exists(KB_cache_path): 
            print(f"Loading cached latent KB from {KB_cache_path}")
            self.kb_repr_mm = np.memmap(
                KB_cache_path,
                mode="r",
                dtype=np.float32,
                shape=(self.n_train, C, D)
            )
            print(f"Latent KB loaded (memmap): shape={(N, D)}")
            return

        # 不存在则创建
        print(f"Encoding training data to latentKB with {self.retriever}...")
        kb_memmap = np.memmap(
            KB_cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(N,C,D)
        )

        offset = 0
        with torch.no_grad():
            # 这里Encoding过程没有加入x_mark。
            for start_idx in tqdm(range(0, N, batch_size), desc="Encoding representations through batches..."):
                end_idx = min(start_idx + batch_size, N)

                batch_data = self.train_data_full[start_idx:end_idx]  # (B, L, C)

                if self.retriever == 'iTransformer' or self.retriever == 'Typology':
                    batch_enc = batch_data  # (B, L, C)
                    enc_output = self.encoder.get_representation(batch_enc) # B, D
                    batch_repr = enc_output.cpu().numpy()    
  
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

    def retrieve_recon(self, x, x_mark, mask):
        """
        根据 Observed 变量的 Representation 检索 KB 中最相似样本，并重构完整序列。
        
        Args:
            x: (B, L, C) —— 输入数据，缺失通道的值应已填 0
            x_mark: (B, L, D_mark) —— 时间特征，必须提供，否则 Embedding 不对齐
            mask: (B, C) —— 1 表示 Observed，0 表示 Missing
        
        Returns:
            x_recon: (B, L, C) —— 重构结果
        """
        device = x.device
        B, L, C = x.shape
        D = self.args.latent_dim

        # ================= 1. 获取 Query Embedding =================
        # 必须使用与训练时完全一致的 get_representation 接口
        # 传入 x_mark 和 mask，触发 TopologyAwareEncoder 的完整逻辑
        with torch.no_grad():
            # query_repr shape: (B, C, D)
            # 这里的 Embedding 已经经过了 Encoder 的"隐式填补"，包含了对缺失通道的推断
            query_repr = self.encoder.get_representation(x, x_mark, mask=mask)

        # 提前归一化 Query，避免在循环中重复计算
        query_norm = F.normalize(query_repr, dim=-1) # (B, C, D)

        # ================= 2. 检索循环 (Batch-wise Retrieval) =================
        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)
        
        # 显存优化参数
        BYTES_PER_FLOAT = 4
        # 显存预算控制 (保留一部分给中间计算)
        MAX_GPU_BYTES = 2 * 1024**3 # 每次只搬运 2GB 到 GPU，防止 OOM
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (C * D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving and calculating simliarity..."):
            end = min(start + kb_batch_size, N)

            # A. 加载 KB Embedding (Key)
            # 从 memmap 读取 (CPU) -> 转 Tensor -> 移到 GPU
            # self.kb_embed_mm 是我们在 prepare_dataset 里存的 (N, C, D)
            kb_chunk_cpu = torch.from_numpy(self.kb_embed_mm[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) 

            # B. 归一化 Key
            kb_chunk_norm = F.normalize(kb_chunk, dim=-1) # (chunk_size, C, D)

            # C. 计算相似度
            # 逻辑：Query(Partial) vs Key(Full)
            # 我们计算所有通道的相似度，因为 Encoder 已经把 Partial 映射到了 Full 的流形上
            # (B, C, D) vs (chunk, C, D) -> (B, chunk, C)
            sim_per_channel = torch.einsum('bcd,ncd->bnc', query_norm, kb_chunk_norm)

            # D. 聚合通道相似度
            # 这里取平均，代表 System State 的整体相似度
            # 也可以改为加权平均 (如果某些通道更重要)
            sim = sim_per_channel.mean(dim=2) # (B, chunk)
            
            # 存入总表
            sim_final[:, start:end] = sim

        # ================= 3. Top-K 选择 =================
        topk = self.topk
        # topk_indices: (B, K)
        topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)
        
        # 计算注意力权重 (Softmax)
        # / temperature 是为了让权重分布更尖锐或平滑，建议沿用训练时的 temp
        prob = F.softmax(topk_sim / self.temperature, dim=1) 

        # ================= 4. 获取原始数据 (Raw Data Retrieval) =================
        # 关键修改：从 self.kb_raw_mm (memmap) 中读取，而不是 train_data_full
        
        # 将 indices 转回 CPU numpy 以便对 memmap 切片
        indices_np = topk_indices.cpu().numpy()
        
        # 这是一个 (B, K, L, C) 的大数组
        # numpy memmap 支持花式索引 (Fancy Indexing)，但如果 B 很大可能会慢
        # retrieved_samples_np = self.kb_raw_mm[indices_np] 
        
        # 优化读取：如果 B 很大，循环读取可能更稳健，防止 RAM 峰值
        retrieved_list = []
        for i in range(B):
            # 读取第 i 个样本对应的 K 个历史片段
            # self.kb_raw_mm: (N, L, C)
            idx = indices_np[i] # (K,)
            sample = self.kb_raw_mm[idx] # (K, L, C)
            retrieved_list.append(sample)
        
        retrieved_samples = torch.tensor(np.stack(retrieved_list), device=device, dtype=torch.float32)

        # ================= 5. 加权融合 (Integration) =================
        # prob: (B, K) -> (B, K, 1, 1)
        weights = prob.view(B, topk, 1, 1)
        
        # 加权求和: (B, L, C)
        # 这里得到的 x_recon 是纯粹从历史数据中合成的完整数据
        x_recon = (weights * retrieved_samples).sum(dim=1)

        
        return x_recon 

    # iTransformer没用到mask/x_mark, 所以效果好？
    def retrieve_recon_old(self, x, observed_mask=None):
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
            # x_obs = x[:, :, obs_idx]  # (B, L, C_obs)

            x_obs = x

            if self.retriever == 'iTransformer' or self.retriever == 'Typology':
                x_query = x_obs  # (B, L, C_obs)
                with torch.no_grad():
                    x_enc = x_query # .permute(0, 2, 1).to(self.device)  # (B, C_obs, L)
                    output = self.encoder.get_representation(
                        x_enc)
                    query_repr = output # (B, C, D)
                    D = query_repr.shape[1]
      
            BYTES_PER_FLOAT = 4

            MAX_GPU_BYTES = 5*1024**3 if self.args.data != 'traffic' else 5*1024**3
            # 40 GB
            kb_batch_size = max(
                1,
                int(MAX_GPU_BYTES // (1 * C * D * BYTES_PER_FLOAT))
            )
            
            N = self.n_train
            sim_final = torch.zeros(B, N, device=device)
            obs_idx = obs_idx.cpu().numpy().tolist()
            for start in tqdm(range(0, N, kb_batch_size)):
                end = min(start + kb_batch_size, N)

                kb_cpu = torch.from_numpy(self.kb_repr_mm[start:end])  # RAM 占用 20G

                # Step 2: 移动到 GPU 并立即归一化
                kb_batch = F.normalize(kb_cpu.to(device), dim=-1)  # GPU 显存: 20G (峰值)
                # (max_batch, C, D)

                query_norm = F.normalize(query_repr, dim=-1)        # (B, C, D)
                # 计算每通道相似度 (B, N, C)
                sim_per_channel = torch.einsum('bcd,ncd->bnc', query_norm, kb_batch)

                # 平均所有通道的相似度 (B, N)
                sim = sim_per_channel.mean(dim=2)
                sim_final[:, start:end] = sim

            # 6. Top-K 检索
            topk = self.topk
            topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)  # (B, K)
            prob = F.softmax(topk_sim / self.temperature, dim=1)  # (B, K)

            # 7. 重构完整序列
            retrieved_samples = self.train_data_full[topk_indices]  # (B, K, L, C)
            weights = prob.view(B, topk, 1, 1)  # (B, K, 1, 1)
            x_recon = (weights * retrieved_samples).sum(dim=1)  # (B, L, C)

            
            return x_recon.to(device)


