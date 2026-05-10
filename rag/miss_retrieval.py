

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
        """ KB  GPU ()"""
        kb_size_gb = self.n_train * self.P * self.D * 4 / 1e9
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if kb_size_gb < gpu_mem_gb * 0.8:
            print(f"🚀 Loading KB to GPU ({kb_size_gb:.2f} GB / {gpu_mem_gb:.1f} GB)")
            self.kb_repr_gpu = torch.from_numpy(
                np.array(self.kb_repr_mm)
            ).to(self.device)
            self.kb_on_gpu = True
            print("✅ KB loaded to GPU")
        else:
            print(f"⚠️ KB too large ({kb_size_gb:.2f} GB) for GPU, keeping on disk")
            self.kb_on_gpu = False


    def training_encoder(self, args, train_loader, vali_loader):
 



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
        train_loader_unshuffled:  shuffle=False  DataLoader！
         Knowledge Base
        """
        task_name = args.task_name
        dataset_name = args.data
        D = args.latent_dim
        N = len(train_loader_unshuffled.dataset)
        C = args.enc_in
        L = args.seq_len
        self.n_train = N

        print(f'Building Retrieval Knowledge Base (N={N}, C={C}, L={L}, D={D})...')


        cache_dir = f"./latentKB/{task_name}/{dataset_name}"
        os.makedirs(cache_dir, exist_ok=True)
        


        embed_path = os.path.join(cache_dir, f"kb_embed_{args.retrieve_encoder}_{args.mask_ratio}_{D}.dat")

        raw_path = os.path.join(cache_dir, f"kb_raw_data_{args.retrieve_encoder}_{args.mask_ratio}_{D}.dat")


        if os.path.exists(embed_path) and os.path.exists(raw_path):
            print(f"Loading cached KB from {cache_dir}...")

            self.kb_embed_mm = np.memmap(embed_path, mode="r", dtype=np.float32, shape=(N, C+args.dim_x_mark, D))
            self.kb_raw_mm = np.memmap(raw_path, mode="r", dtype=np.float32, shape=(N, L, C))
            return


        print("Constructing new KB...")
        fp_embed = np.memmap(embed_path, mode="w+", dtype=np.float32, shape=(N, C+args.dim_x_mark, D))
        fp_raw = np.memmap(raw_path, mode="w+", dtype=np.float32, shape=(N, L, C))

        self.encoder.to(self.device)
        self.encoder.eval()

        offset = 0
        with torch.no_grad():


            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_x_full) in tqdm(enumerate(train_loader_unshuffled), total=len(train_loader_unshuffled)):
                


                

                # x_full: [B, L, C]
                x_full = batch_x_full.float().to(self.device)
                
                # x_mark: [B, L, D_mark]
                if batch_x_mark is not None:
                    x_mark = batch_x_mark.float().to(self.device)
                



                B_current = x_full.shape[0]
                full_mask = torch.ones(B_current, C, device=self.device)


                batch_raw_np = x_full.cpu().numpy()
                fp_raw[offset : offset + B_current] = batch_raw_np


                if self.retriever == 'Typology' or self.retriever == 'iTransformer':


                    enc_output = self.encoder.get_representation(x_full, x_mark, mask=full_mask) 

                    
                    batch_embed_np = enc_output.cpu().numpy()
                    fp_embed[offset : offset + B_current] = batch_embed_np
                
                else:

                    pass

                offset += B_current
        

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


        print(f"Encoding training data to latentKB with {self.retriever}...")
        kb_memmap = np.memmap(
            KB_cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(N,C,D)
        )

        offset = 0
        with torch.no_grad():

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



        self.kb_repr_mm = np.memmap(
            KB_cache_path,
            mode="r",
            dtype=np.float32,
            shape=(N, C, D)
        )

        print(f"Latent KB saved safely to {KB_cache_path}")

    def retrieve_recon(self, x, x_mark, mask):
        """
         Observed  Representation  KB ，
        
        Args:
            x: (B, L, C) —— ， 0
            x_mark: (B, L, D_mark) —— ，， Embedding 
            mask: (B, C) —— 1  Observed，0  Missing
        
        Returns:
            x_recon: (B, L, C) —— 
        """
        device = x.device
        B, L, C = x.shape
        D = self.args.latent_dim




        with torch.no_grad():
            # query_repr shape: (B, C, D)

            query_repr = self.encoder.get_representation(x, x_mark, mask=mask)


        query_norm = F.normalize(query_repr, dim=-1) # (B, C, D)


        N = self.n_train
        sim_final = torch.zeros(B, N, device=device)
        

        BYTES_PER_FLOAT = 4

        MAX_GPU_BYTES = 2 * 1024**3
        kb_batch_size = max(1, int(MAX_GPU_BYTES // (C * D * BYTES_PER_FLOAT)))

        for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving and calculating simliarity..."):
            end = min(start + kb_batch_size, N)




            kb_chunk_cpu = torch.from_numpy(self.kb_embed_mm[start:end]) 
            kb_chunk = kb_chunk_cpu.to(device) 


            kb_chunk_norm = F.normalize(kb_chunk, dim=-1) # (chunk_size, C, D)




            # (B, C, D) vs (chunk, C, D) -> (B, chunk, C)
            sim_per_channel = torch.einsum('bcd,ncd->bnc', query_norm, kb_chunk_norm)




            sim = sim_per_channel.mean(dim=2) # (B, chunk)
            

            sim_final[:, start:end] = sim


        topk = self.topk
        # topk_indices: (B, K)
        topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)
        


        prob = F.softmax(topk_sim / self.temperature, dim=1) 



        

        indices_np = topk_indices.cpu().numpy()
        


        # retrieved_samples_np = self.kb_raw_mm[indices_np] 
        

        retrieved_list = []
        for i in range(B):

            # self.kb_raw_mm: (N, L, C)
            idx = indices_np[i] # (K,)
            sample = self.kb_raw_mm[idx] # (K, L, C)
            retrieved_list.append(sample)
        
        retrieved_samples = torch.tensor(np.stack(retrieved_list), device=device, dtype=torch.float32)


        # prob: (B, K) -> (B, K, 1, 1)
        weights = prob.view(B, topk, 1, 1)
        


        x_recon = (weights * retrieved_samples).sum(dim=1)

        
        return x_recon 


    def retrieve_recon_old(self, x, observed_mask=None):
            """
             observed representation (P, D) KB(P, D)，
            
            Args:
                x: (B, L, C) —— ，missing （ observed）
                observed_mask: (C,) —— 1  observed，0  missing
            
            Returns:
                x_recon: (B, L, C) —— （missing ，observed ）
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

                kb_cpu = torch.from_numpy(self.kb_repr_mm[start:end])


                kb_batch = F.normalize(kb_cpu.to(device), dim=-1)
                # (max_batch, C, D)

                query_norm = F.normalize(query_repr, dim=-1)        # (B, C, D)

                sim_per_channel = torch.einsum('bcd,ncd->bnc', query_norm, kb_batch)


                sim = sim_per_channel.mean(dim=2)
                sim_final[:, start:end] = sim


            topk = self.topk
            topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)  # (B, K)
            prob = F.softmax(topk_sim / self.temperature, dim=1)  # (B, K)


            retrieved_samples = self.train_data_full[topk_indices]  # (B, K, L, C)
            weights = prob.view(B, topk, 1, 1)  # (B, K, 1, 1)
            x_recon = (weights * retrieved_samples).sum(dim=1)  # (B, L, C)

            
            return x_recon.to(device)


