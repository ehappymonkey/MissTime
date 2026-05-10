






import torch
import torch.nn as nn
import torch.nn.functional as F



from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, TopologyAwareEncoder


import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm

class iTransformerContrastive(nn.Module):
    """
    Missing-Aware Retriever based on iTransformer Encoder.
    Goal: Align partial-view embeddings with complete-view embeddings via Contrastive Learning.
    """
    def __init__(self, configs, device='cpu'):
        super(iTransformerContrastive, self).__init__()
        self.seq_len = configs.seq_len

        #configs.d_model = 512
        self.d_model = configs.latent_dim
        self.temperature = configs.temperature if hasattr(configs, 'temperature') else 0.07
        
        # 1. Inverted Embedding: [B, L, C] -> [B, C, D]

        if configs.retrieve_encoder == 'Typology':
            self.enc_embedding = TopologyAwareEncoder(configs)
        else:
            self.enc_embedding = DataEmbedding_inverted(
                configs.seq_len, 
                self.d_model, 
                configs.embed, 
                configs.freq,
                configs.dropout
            )

        # 2. Encoder: Process interactions between variables (tokens)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), 
                        self.d_model, 
                        configs.n_heads
                    ),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )




        self.projector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model) 
        )
        self.device = device
        self.to(self.device)
        self.args = configs


    def get_representation(self, x, x_mark=None, mask=None):
        """
        
        x: [B, L, C]
        x_mark: [B, L, D_mark] ()
        mask: [B, C] (1 for observed, 0 for missing) -  C 
        """
        if mask == None:
            mask = torch.ones(x.shape[0], x.shape[2], device=x.device, dtype=torch.float32)
        enc_out = self.enc_embedding(x, x_mark.to(x.device) if x_mark is not None else None, mask=mask.to(x.device))

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        z = self.projector(enc_out)
        
        return z


    def compute_contrastive_loss(self, z_anchor, z_positive):
        B, C, D = z_anchor.shape
        

        z_anchor_flat = z_anchor.view(B, -1)  # [B, C*D]
        z_positive_flat = z_positive.view(B, -1)  # [B, C*D]
        

        z_anchor_flat = F.normalize(z_anchor_flat, dim=1)
        z_positive_flat = F.normalize(z_positive_flat, dim=1)
        

        logits = torch.matmul(z_anchor_flat, z_positive_flat.T) / self.temperature
        labels = torch.arange(B, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute_hard_negative_loss(self, z_partial, z_complete, temp=0.07, mining_temp=0.1):
        """
         Flatten (System State)  Hard Negative Mining 
        
        Args:
            z_partial: [B, C, D] or [B, C*D]
            z_complete: [B, C, D] or [B, C*D]
        """
        B = z_partial.shape[0]
        

        z_p_flat = z_partial.view(B, -1)     # [B, C*D]
        z_c_flat = z_complete.view(B, -1)    # [B, C*D]
        
        z_p_norm = F.normalize(z_p_flat, dim=1)
        z_c_norm = F.normalize(z_c_flat, dim=1)
        
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        


        sim_pp = torch.matmul(z_p_norm, z_p_norm.T) # [B, B]
        



        sim_cc = torch.matmul(z_c_norm, z_c_norm.T) # [B, B]
        


        logits = torch.matmul(z_p_norm, z_c_norm.T) / temp # [B, B]
        
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        
        mask_diag = torch.eye(B, device=z_partial.device)
        

        hardness_term = torch.exp(sim_pp / mining_temp)
        




        divergence_term = (1.0 - sim_cc).clamp(min=0.0)
        

        weights = (hardness_term * divergence_term).detach()
        

        weights = weights * (1 - mask_diag)
        
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        

        exp_logits = torch.exp(logits)
        
        # sum_{k != i} ( w_{ik} * exp(sim) )
        neg_sum = (weights * exp_logits).sum(dim=1) 
        

        pos_term = exp_logits.diag()
        
        # Loss
        loss = -torch.log(pos_term / (pos_term + neg_sum + 1e-9)).mean()
        
        return loss

    def compute_hard_negative_loss_mean(self, z_partial, z_complete, temp=0.07, mining_temp=0.1):
        """
        Modified to use Channel-wise Average Similarity.
        
        Args:
            z_partial: [B, C, D] (： Flatten， C )
            z_complete: [B, C, D]
            temp: InfoNCE 
            mining_temp:  Hard Negative 
        """
        B, C, D = z_partial.shape
        


        z_p_norm = F.normalize(z_partial, dim=-1) # [B, C, D]
        z_c_norm = F.normalize(z_complete, dim=-1) # [B, C, D]
        
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        


        # i: batch idx 1, j: batch idx 2, c: channel, d: dimension




        
        sim_obs = torch.einsum('icd, jcd -> ij', z_p_norm, z_p_norm) / C  # [B, B]
        

        logits = torch.einsum('icd, jcd -> ij', z_p_norm, z_c_norm) / C   # [B, B]
        

        logits = logits / temp

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        

        mask_diag = torch.eye(B, device=z_partial.device)
        

        weights = torch.exp(sim_obs / mining_temp).detach()
        weights = weights * (1 - mask_diag)
        

        exp_logits = torch.exp(logits)
        neg_sum = (weights * exp_logits).sum(dim=1) # [B]
        

        pos_term = exp_logits.diag() # [B]
        
        # 7. Final Loss
        loss = -torch.log(pos_term / (pos_term + neg_sum + 1e-9)).mean()
        
        return loss
    

    def forward(self, batch_x, batch_x_full, batch_mask, x_mark=None):
        """
        Training Step:
        batch_x: [B, L, C] (，0)
        batch_x_full: [B, L, C] ()
        batch_mask: [B, C] (1，0)
        """

        z_partial = self.get_representation(batch_x, x_mark, mask=batch_mask)
        
        batch_mask_full = torch.ones_like(batch_mask)
        z_full = self.get_representation(batch_x_full, x_mark, mask=batch_mask_full)

        if self.args.contrastive_loss == 'hard_negative':
            loss_p2f = self.compute_hard_negative_loss(z_partial, z_full)
            loss_f2p = self.compute_hard_negative_loss(z_full, z_partial)
        else:
            loss_p2f = self.compute_contrastive_loss(z_partial, z_full)
            loss_f2p = self.compute_contrastive_loss(z_full, z_partial)
        
        total_loss = (loss_p2f + loss_f2p) / 2
        return total_loss

    def fit(self, train_loader, vali_loader=None, path=None):
        """
        
        """
        # if path is not None:
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        

        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        
        use_amp = self.args.use_amp if hasattr(self.args, 'use_amp') else True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        patience = 10       
        counter = 0    

        best_val_loss = float('inf')
        best_checkpoints = None
        for epoch in range(self.args.encoder_epochs):
            self.train()
            total_train_loss = 0

            for i, batch in tqdm(enumerate(train_loader)):
                if self.args.task_name == 'long_term_forecast':
                    batch_x_full = batch[6]
                    batch_x_mark = batch[3]
                    batch_mask = batch[5]
                    batch_x = batch[1]
                    
                elif self.args.task_name == 'imputation':
                    batch_x_full = batch[5]
                    batch_x_mark = batch[2]
                    batch_mask = batch[4]
                    batch_x = batch[0]
                
                elif self.args.task_name == 'anomaly_detection' or self.args.task_name == 'classification':
                    batch_x_full = batch[1]
                    batch_x_mark = None
                    batch_mask = batch[3]
                    batch_x = batch[0]
 
            # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(train_loader):
        
                batch_x = batch_x.float().to(self.device)
                batch_x_full = batch_x_full.float().to(batch_x.device)
                batch_mask = batch_mask.unsqueeze(0).expand(batch_x.shape[0], -1)
                batch_mask = batch_mask.float().to(batch_x.device)
                batch_x_mark = batch_x_mark.float().to(batch_x.device) if batch_x_mark is not None else None
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = self.forward(batch_x, batch_x_full, batch_mask, batch_x_mark)
                
                if use_amp:
                    # Scale loss -> Backward -> Unscale -> Step -> Update Scaler
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
     
                
                total_train_loss += loss.item()
                # print({"train_loss": f"{loss.item():.4f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.6f}")
            

            if vali_loader is not None:
                self.eval()
                total_val_loss = 0
                with torch.no_grad():

                    for i, batch in tqdm(enumerate(vali_loader)):
                        if self.args.task_name == 'long_term_forecast':
                            batch_x_full = batch[6]
                            batch_x_mark = batch[3]
                            batch_mask = batch[5]
                            batch_x = batch[1]
                            
                        elif self.args.task_name == 'imputation':
                            batch_x_full = batch[5]
                            batch_x_mark = batch[2]
                            batch_mask = batch[4]
                            batch_x = batch[0]

                        elif self.args.task_name == 'anomaly_detection' or self.args.task_name == 'classification':
                            batch_x_full = batch[1]
                            batch_x_mark = None
                            batch_mask = batch[3]
                            batch_x = batch[0]
            

                    # for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(vali_loader):
                        batch_x = batch_x.float().to(self.device)
                        batch_x_full = batch_x_full.float().to(batch_x.device)
                        batch_mask = batch_mask.unsqueeze(0).expand(batch_x.shape[0], -1)
                        batch_mask = batch_mask.float().to(batch_x.device)
                        batch_x_mark = batch_x_mark.float().to(batch_x.device) if batch_x_mark is not None else None
                        
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            val_loss = self.forward(batch_x, batch_x_full, batch_mask, batch_x_mark)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(vali_loader)
                print(f"Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_checkpoints = self.state_dict()
                    counter = 0
                    print(f'--> Best model updated at epoch {epoch+1}!')
                else:
                    counter += 1
                    print(f'--> EarlyStopping counter: {counter} out of {patience}')
                    if counter >= patience:
                        print("Early stopping triggered. Training stopped.")
                        break
            else: 
                best_checkpoints = self.state_dict()
            

            scheduler.step()
        self.load_state_dict(best_checkpoints)
        print("🎉 Training completed!")
        return self


if __name__ == "__main__":

    class Configs:
        seq_len = 96
        d_model = 128
        embed = 'timeF'
        freq = 'h'
        dropout = 0.1
        factor = 1
        n_heads = 4
        d_ff = 256
        activation = 'gelu'
        e_layers = 2
        temperature = 0.07

    configs = Configs()
    model = iTransformerContrastive(configs)
    

    B, L, C = 32, 96, 7
    batch_x_full = torch.randn(B, L, C)
    

    batch_mask = torch.randint(0, 2, (B, C)).float() # [B, C], 0 or 1

    batch_mask[:, 0] = 1.0 
    

    batch_x = batch_x_full * batch_mask.unsqueeze(1)
    

    loss = model(batch_x, batch_x_full, batch_mask)
    print(f"Contrastive Loss: {loss.item()}")


    model.eval()
    with torch.no_grad():
        # Query Embedding (Partial)
        query_emb = model.get_representation(batch_x, None, batch_mask)
        # Key Embedding (Full)
        key_emb = model.get_representation(batch_x_full, None, None)
        

        query_emb = F.normalize(query_emb, dim=1)
        key_emb = F.normalize(key_emb, dim=1)
        sim = torch.matmul(query_emb, key_emb.T)
        print(f"Similarity Matrix Shape: {sim.shape}") # [32, 32]