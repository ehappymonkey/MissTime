




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

    def prepare_dataset(self, train_data, task_name=None, dataset_name=None, batch_size=4):
        print('Preparing Retrieval Dataset...')

        # --------------------------------------------------

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


        print("🔍 Probing model output shape with a single sample...")
        with torch.no_grad():
            sample = self.train_data_full[:1].permute(0, 2, 1)
            sample_out = self.encoder(x_enc=sample.to(self.device), reduction="none")
            _, C_enc, P, D = sample_out.embeddings.shape
        print(f"Detected: P={P}, D={D} (C={C_enc} will be pooled)")

        self.P = P
        self.D = D


        cache_dir = f"./latentKBCache/{task_name}/{dataset_name}"
        os.makedirs(cache_dir, exist_ok=True)
        KB_cache_path = os.path.join(cache_dir, f"repr_KB_pooled_N{N}_P{P}_D{D}.dat")


        if os.path.exists(KB_cache_path):
            print(f"Loading cached POOLED latent KB from {KB_cache_path}")
            self.kb_repr_mm = np.memmap(
                KB_cache_path,
                mode="r",
                dtype=np.float32,
                shape=(N, P, D)
            )
            print(f"Loaded pooled KB: shape={self.kb_repr_mm.shape}")
            return


        print("Encoding and pooling training data with MOMENT...")
        

        kb_memmap = np.memmap(
            KB_cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(N, P, D)
        )

        offset = 0
        with torch.no_grad():
            for start_idx in tqdm(range(0, N, batch_size), desc="Encoding batches"):
                end_idx = min(start_idx + batch_size, N)
                

                batch_data = self.train_data_full[start_idx:end_idx]
                batch_enc = batch_data.permute(0, 2, 1).to(self.device)
                enc_output = self.encoder(x_enc=batch_enc, reduction="none")
                

                batch_pooled = enc_output.embeddings.mean(dim=1)  # (B, P, D)
                batch_pooled = F.normalize(batch_pooled, dim=-1)
                

                kb_memmap[offset:offset + batch_pooled.shape[0]] = batch_pooled.cpu().numpy()
                offset += batch_pooled.shape[0]
                
                del batch_enc, enc_output, batch_pooled
                torch.cuda.empty_cache()


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
        

    #     with torch.no_grad():
    #         sample = self.train_data_full[:1].permute(0, 2, 1)
    #         sample_out = self.encoder(x_enc=sample, reduction="none")
    #         _, C_enc, P, D = sample_out.embeddings.shape

    #     print(f"Latent repr shape per sample: (C={C_enc}, P={P}, D={D})")

    #     # --------------------------------------------------

    #     # --------------------------------------------------
    #     cache_dir = f"./latentKBCache/{task_name}/{dataset_name}"
    #     os.makedirs(cache_dir, exist_ok=True)


    #     KB_cache_path = os.path.join(cache_dir, f"repr_KB_{N}.dat")
    #     # meta_path = os.path.join(cache_dir, f"repr_KB_{N}_meta.pt")

    #     # --------------------------------------------------

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

    #     # --------------------------------------------------
    #     print("Encoding training data with MOMENT...")

    

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



    #             kb_memmap[offset:offset + batch_repr.shape[0]] = batch_repr
    #             offset += batch_repr.shape[0]

    #             del batch_enc, enc_output, batch_repr
    #             torch.cuda.empty_cache()



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
        

        obs_idx = torch.where(observed_mask)[0]
        x_obs = x[:, :, obs_idx]
        x_query = x_obs.permute(0, 2, 1).to(self.device)
        
        with torch.no_grad():
            output = self.encoder(x_enc=x_query, reduction='none')
            query_repr = output.embeddings  # (B, C_obs, P, D)
        

        query_pooled = query_repr.mean(dim=1)  # (B, P, D)
        query_norm = F.normalize(query_pooled, dim=-1)  # (B, P, D)


        if hasattr(self, 'kb_on_gpu') and self.kb_on_gpu:

            sim_final = torch.einsum('bpd, npd -> bnp', query_norm, self.kb_repr_gpu)
            sim_final = sim_final.mean(dim=2)
        else:

            N = self.n_train
            P, D = query_norm.shape[1], query_norm.shape[2]
            

            MAX_GPU_BYTES = 40 * 1024**3
            kb_batch_size = max(1, int(MAX_GPU_BYTES // (P * D * 4)))



            sim_final = torch.zeros(B, N, device=device)
            

            # for start in tqdm(range(0, N, kb_batch_size), desc="Retrieving from KB..."):
            for start in range(0, N, kb_batch_size):
                end = min(start + kb_batch_size, N)
                

                kb_batch = torch.from_numpy(
                    self.kb_repr_mm[start:end]  # 👈 (batch_size, P, D)
                ).to(device)
                

                sim = torch.einsum('bpd, npd -> bnp', query_norm, kb_batch)  # (B, batch_size)
                sim_final[:, start:end] = sim


        topk = min(self.topk, self.n_train)
        topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)
        prob = F.softmax(topk_sim / self.temperature, dim=1)
        
        retrieved_samples = self.train_data_full[topk_indices]
        weights = prob.view(B, topk, 1, 1)
        x_recon = (weights * retrieved_samples).sum(dim=1)
        
        return x_recon.to(device)
        
    # def retrieve_recon(self, x, observed_mask=None):
    #     """

        
    #     Args:


        
    #     Returns:

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

    #     for start in tqdm(range(0, N, kb_batch_size), desc="Quering Similarity through KB batches..."):
    #         end = min(start + kb_batch_size, N)

    #         kb_batch = torch.from_numpy(
    #             self.kb_repr_mm[start:end][:, obs_idx, :, :]
    #         ).to(device)       

    #         kb_pooled = kb_batch.mean(dim=1)        # (max_batch, P, D) 
    #         kb_norm = F.normalize(kb_pooled, dim=-1)        # (max_batch, P, D)
        

    #         sim = torch.einsum('bpd,npd->bnp', query_norm, kb_norm)

    #         sim_chunks.append(sim)


    #     topk = min(self.topk, self.n_train)
    #     sim_final = torch.cat(sim_chunks, dim=1)  # (B, N)
    #     topk_sim, topk_indices = torch.topk(sim_final, topk, dim=1)  # (B, K)
    #     prob = F.softmax(topk_sim / self.temperature, dim=1)  # (B, K)


    #     retrieved_samples = self.train_data_full[topk_indices]  # (B, K, L, C)
    #     weights = prob.view(B, topk, 1, 1)  # (B, K, 1, 1)
    #     x_recon = (weights * retrieved_samples).sum(dim=1)  # (B, L, C)

        
    #     return x_recon.to(device)

