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
        
        self.kb_embed = None     
        self.kb_raw = None        
        
        self.n_train = 0
        self.retriever = retriever
        self.encoder = None
        self.device = device
        self.args = configs
        

    def training_encoder(self, args, train_loader, vali_loader):


        # forecasting logic...
        # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(train_loader):
        args.dim_x_mark = 0 
        
        model = iTransformerContrastive(args, self.device) 
        path = f"./pretrained_encoder/{args.task_name}/{args.data}/{args.retrieve_encoder}_{args.contrastive_loss}_{args.mask_ratio}_{model.d_model}_checkpoints.pt"

        if os.path.exists(path):
            print(f"Loading pretrained encoder from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Start training Our own {args.retrieve_encoder} Retrieval Encoder..., Contrastive Loss: {args.contrastive_loss}")
            model = model.fit(train_loader, vali_loader=vali_loader, path=path)
            print(f"Training of Retrieval Encoder finished.")

        self.encoder = model

    def prepare_dataset(self, args, train_loader_unshuffled):

        task_name = args.task_name
        dataset_name = args.data
        D = args.latent_dim
        N = len(train_loader_unshuffled.dataset) 
        C = args.enc_in 
        L = args.seq_len
        self.n_train = N

        print(f'Building Retrieval Knowledge Base (N={N}, C={C}, L={L}, D={D})...')

    
        print("Constructing new KB in RAM...")
        
        # 直接在内存分配 (RAM)
        self.kb_embed = np.zeros((N, C+args.dim_x_mark, D), dtype=np.float32)
        self.kb_raw = np.zeros((N, L, C), dtype=np.float32)

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

                self.kb_raw[offset : offset + B_current] = x_full.cpu().numpy()

                if self.retriever in ['Typology', 'iTransformer']: 
                    enc_output = self.encoder.get_representation(x_full, x_mark, mask=full_mask) 
                    self.kb_embed[offset : offset + B_current] = enc_output.cpu().numpy()
                else:
                    pass

                offset += B_current
        
        print("KB encoding finished.")



    def retrieve_recon_whole(self, x, x_mark, mask):
        
        device = x.device
        B, L, C = x.shape
        D = self.args.latent_dim


        with torch.no_grad():
            query_repr = self.encoder.get_representation(x, x_mark, mask=mask) # [B, C, D]

        # 1. Flatten: [B, C, D] -> [B, C*D]
        query_flat = query_repr.reshape(B, -1)
        
        query_global_norm = F.normalize(query_flat, p=2, dim=1) 

        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)


        BYTES_PER_FLOAT = 4
        MAX_GPU_BYTES = 2 * 1024**3 
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (C * D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving..."):
            end = min(start + kb_batch_size, N)

            # 加载 KB Chunk (CPU -> GPU)
            kb_chunk_cpu = torch.from_numpy(self.kb_embed[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) # [Chunk, C, D]

            # 1. Flatten Key: [Chunk, C, D] -> [Chunk, C*D]
            kb_flat = kb_chunk.reshape(kb_chunk.shape[0], -1)
            
            # 2. Global Normalize Key
            kb_global_norm = F.normalize(kb_flat, p=2, dim=1)

            # [B, C*D] @ [Chunk, C*D].T -> [B, Chunk]
            sim = torch.matmul(query_global_norm, kb_global_norm.T)
            
            sim_final[:, start:end] = sim

        # 3. Top-K Selection
        topk_sim, topk_indices = torch.topk(sim_final, self.topk, dim=1)
        prob = F.softmax(topk_sim / self.temperature, dim=1) 

        # 4. Raw Data Retrieval (From RAM)
        indices_np = topk_indices.cpu().numpy() # (B, K)
        

        retrieved_samples_np = self.kb_raw[indices_np] # Result: (B, K, L, C)

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
        
        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)
        

        BYTES_PER_FLOAT = 4
        MAX_GPU_BYTES = 2 * 1024**3 
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (C * D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving and calculating sim...", leave=False):
            end = min(start + kb_batch_size, N)

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
        indices_np = topk_indices.cpu().numpy() # (B, K)
        
        retrieved_samples_np = self.kb_raw[indices_np] # Result: (B, K, L, C)
        

        retrieved_samples = torch.from_numpy(retrieved_samples_np).to(device)
        weights = prob.view(B, self.topk, 1, 1)
        x_recon = (weights * retrieved_samples).sum(dim=1)

        return x_recon