import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os

class Retrieval():
    def __init__(self, configs, topk=10, temperature=0.1, device='cpu', retriever='RawFeature'):
        self.topk = topk
        self.temperature = temperature
        
        self.kb_raw = None        
        
        self.n_train = 0
        self.retriever = retriever 
        self.device = device
        self.args = configs
        

    def prepare_dataset(self, args, train_loader_unshuffled):

        task_name = args.task_name
        dataset_name = args.data
        N = len(train_loader_unshuffled.dataset) 
        C = args.enc_in 
        L = args.seq_len
        self.n_train = N

        print(f'Building Raw Feature Knowledge Base (N={N}, C={C}, L={L})...')
        print("Loading raw data into RAM...")
        

        self.kb_raw = np.zeros((N, L, C), dtype=np.float32)

        offset = 0
        with torch.no_grad():
            # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_x_full) in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):
            for i, batch in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):

                if args.task_name == 'long_term_forecast':
                    batch_x_full = batch[6]
                elif args.task_name == 'imputation':
                    batch_x_full = batch[4]
                elif args.task_name == 'classification' or args.task_name == 'anomaly_detection':
                    batch_x_full = batch[1]


                x_full = batch_x_full.float() 
                B_current = x_full.shape[0]
                self.kb_raw[offset : offset + B_current] = x_full.numpy()

                offset += B_current
        
        print("Raw Knowledge Base built.")


    def retrieve_recon(self, x):

        device = x.device
        B, L, C = x.shape
        
        
        query_data = x
        query_flat = query_data.reshape(B, -1)
        query_global_norm = F.normalize(query_flat, p=2, dim=1) 


        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)


        BYTES_PER_FLOAT = 4
        MAX_GPU_BYTES = 2 * 1024**3 
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (L * C * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving (Raw Feature)...", leave=False):
            end = min(start + kb_batch_size, N)

            # self.kb_raw 是 [N, L, C]
            kb_chunk_cpu = torch.from_numpy(self.kb_raw[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) # [Chunk, L, C]

            # [Chunk, L, C] -> [Chunk, L*C]
            kb_flat = kb_chunk.reshape(kb_chunk.shape[0], -1)
            
            # Global Normalize Key
            kb_global_norm = F.normalize(kb_flat, p=2, dim=1)

            # Query [B, D] @ Key_T [D, Chunk] -> [B, Chunk] (D = L*C)
            sim = torch.matmul(query_global_norm, kb_global_norm.T)
            
            sim_final[:, start:end] = sim

        # ================= 3. Top-K Selection =================
        # topk_indices: (B, K)
        topk_sim, topk_indices = torch.topk(sim_final, self.topk, dim=1)
        
        prob = F.softmax(topk_sim / self.temperature, dim=1) 

        indices_np = topk_indices.cpu().numpy() # (B, K)
        
        retrieved_samples_np = self.kb_raw[indices_np] # Result: (B, K, L, C)
        
        retrieved_samples = torch.from_numpy(retrieved_samples_np).to(device)

        # weights: (B, K, 1, 1)
        weights = prob.view(B, self.topk, 1, 1)
        
        x_recon = (weights * retrieved_samples).sum(dim=1) # (B, L, C)
        
        return x_recon