
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os
import argparse

class Retrieval():
    def __init__(self, configs, topk=10, temperature=0.1, device='cpu', retriever='TimerXL'):
        self.topk = topk
        self.temperature = temperature
        self.device = device
        self.args = configs
        self.retriever = retriever

        self.kb_embed = None      # (N, C, D) D=1024 for TimerXL
        self.kb_raw = None        # (N, L, C)
        self.n_train = 0
        
        self.encoder = None
        self.latent_dim = 0    


        if self.retriever == 'TimerXL':
            print("Loading Pre-trained TimerXL as Encoder...")
            from models import timer_xl  
            
            xl_args = argparse.Namespace()
            xl_args.input_token_len = 96     
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
   
            model = timer_xl.Model(xl_args)
            

            ckpt_path = '/home/zz/yuyuan/rag/checkpoint.pth'
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"✅ TimerXL checkpoint loaded from {ckpt_path}")
            else:
                print(f"⚠️ Warning: Checkpoint {ckpt_path} not found! Using random init.")

            self.encoder = model
            self.encoder.to(self.device)
            self.encoder.eval() 
        
            self.latent_dim = xl_args.d_model # 1024
        else:
            raise NotImplementedError(f"Retriever {retriever} not implemented yet.")
        

    def prepare_dataset(self, args, train_loader_unshuffled):

        N = len(train_loader_unshuffled.dataset) 
        C = args.enc_in 
        L = args.seq_len
        D = self.latent_dim # 1024
        self.n_train = N

        print(f'Building TimerXL Knowledge Base (Mean-Pooled) (N={N}, D={D})...')

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
                elif args.task_name == 'classification' or args.task_name == 'anomaly_detection':
                    batch_x_full = batch[1]


                # batch_x_full: (B, L, C)
                batch_data = batch_x_full.float().to(self.device)
                B_current = batch_data.shape[0]

                # 1. 存 Raw Data (CPU)
                self.kb_raw[offset : offset + B_current] = batch_data.cpu().numpy()


                if self.retriever == 'TimerXL':
                    batch_repr = self.encoder.get_embedding(batch_data) 
                    batch_repr_mean = batch_repr.mean(dim=1)

                    self.kb_embed[offset : offset + B_current] = batch_repr_mean.cpu().numpy()

                offset += B_current
        
        print("✅ TimerXL Knowledge Base built and stored in RAM.")


    def retrieve_recon(self, x):

        device = x.device
        B, L, C = x.shape
        D = self.latent_dim


        self.encoder.eval()
        with torch.no_grad():
            query_repr = self.encoder.get_embedding(x).mean(dim=1)


        query_norm = F.normalize(query_repr, p=2, dim=1) 

        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)


        BYTES_PER_FLOAT = 4
        MAX_GPU_BYTES = 2 * 1024**3 # 2GB
        kb_batch_size = max(1, int(MAX_GPU_BYTES // ( D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving (TimerXL)...", leave=False):
            end = min(start + kb_batch_size, N)


            kb_chunk_cpu = torch.from_numpy(self.kb_embed[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) # [Chunk, D]

            
            kb_chunk_norm = F.normalize(kb_chunk, p=2, dim=1)

            # [B, C*D] @ [Chunk, C*D].T -> [B, Chunk]
            sim = torch.matmul(query_norm, kb_chunk_norm.T)
            
            sim_final[:, start:end] = sim

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