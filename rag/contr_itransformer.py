# 技术上要做的 a. 改C b. mask掉存在对于缺失的Attention。


# 学习率大，batch小， 超参数原因等导致对比学习验证损失大；
# hard neg 代码或者超参数原因导验证损失很大，而且不work；
# 跑了ele, wea, tra, pems数据集看效果。 2:52.

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设这些模块已经在你的路径下可引用
# 如果没有，请确保将 iTransformer 的 layers 文件夹放在同级目录
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
        # self.d_model = configs.d_model # Encoder的dim, 默认512。
        #configs.d_model = 512
        self.d_model = configs.latent_dim
        self.temperature = configs.temperature if hasattr(configs, 'temperature') else 0.07
        
        # 1. Inverted Embedding: [B, L, C] -> [B, C, D]
        # 这将每个变量(Channel)的整个时间序列映射为一个Token
        if configs.retrieve_encoder == 'Typology':
            self.enc_embedding = TopologyAwareEncoder(configs) # 里面有dim_x_mark
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

        # 3. Projection Head (关键组件)
        # 将 Backbone 的表示映射到 Metric Space，通常使用 2-layer MLP
        # 训练完成后，检索时通常使用 Project 之前的 representation，或者 Project 之后的 (视实验效果定)
        self.projector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model) 
        )
        self.device = device
        self.to(self.device)
        self.args = configs

    # 输出表示 B,C,D。这里应该分masked 的输出和全部的输出！
    def get_representation(self, x, x_mark=None, mask=None):
        """
        提取特征表示的核心函数
        x: [B, L, C]
        x_mark: [B, L, D_mark] (可选)
        mask: [B, C] (1 for observed, 0 for missing) - 对应 C 维度
        """
        if mask == None: # 建立数据库时候输入mask全部为1.
            mask = torch.ones(x.shape[0], x.shape[2], device=x.device, dtype=torch.float32)
        enc_out = self.enc_embedding(x, x_mark.to(x.device) if x_mark is not None else None, mask=mask.to(x.device)) # 没有加x_mark.

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        z = self.projector(enc_out)
        
        return z


    def compute_contrastive_loss(self, z_anchor, z_positive):
        B, C, D = z_anchor.shape
        
        # 1. 通道拼接 (保留通道顺序信息)
        z_anchor_flat = z_anchor.view(B, -1)  # [B, C*D]
        z_positive_flat = z_positive.view(B, -1)  # [B, C*D]
        
        # 2. L2 归一化 (关键！)
        z_anchor_flat = F.normalize(z_anchor_flat, dim=1)
        z_positive_flat = F.normalize(z_positive_flat, dim=1)
        
        # 3. 标准 InfoNCE
        logits = torch.matmul(z_anchor_flat, z_positive_flat.T) / self.temperature
        labels = torch.arange(B, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute_hard_negative_loss(self, z_partial, z_complete, temp=0.07, mining_temp=0.1):
        """
        基于 Flatten (System State) 的 Hard Negative Mining 对比学习。
        
        Args:
            z_partial: [B, C, D] or [B, C*D]
            z_complete: [B, C, D] or [B, C*D]
        """
        B = z_partial.shape[0]
        
        # 1. Flatten & Normalize (体现 System State 整体观)
        z_p_flat = z_partial.view(B, -1)     # [B, C*D]
        z_c_flat = z_complete.view(B, -1)    # [B, C*D]
        
        z_p_norm = F.normalize(z_p_flat, dim=1)
        z_c_norm = F.normalize(z_c_flat, dim=1)
        
        # -----------------------------------------------------------
        # 2. 计算相似度矩阵 (全部基于 Global State Similarity)
        # -----------------------------------------------------------
        
        # A. 观测相似度 (Looks like?): Partial vs Partial
        # 用于判断是否是"假邻居"的嫌疑人
        sim_pp = torch.matmul(z_p_norm, z_p_norm.T) # [B, B]
        
        # B. 状态相似度 (Actually is?): Complete vs Complete
        # 用于判断是否是"真兄弟" (False Negative)
        # 注意：这里用 Complete 计算更准确，因为它是 Oracle
        sim_cc = torch.matmul(z_c_norm, z_c_norm.T) # [B, B]
        
        # C. 目标对齐度 (Alignment): Partial vs Complete
        # 这是 Loss 的 Logits
        logits = torch.matmul(z_p_norm, z_c_norm.T) / temp # [B, B]
        
        # -----------------------------------------------------------
        # 3. 构造动态权重 (Dynamics-Aware Hardness)
        # -----------------------------------------------------------
        
        mask_diag = torch.eye(B, device=z_partial.device)
        
        # Term 1: Hardness (观测越像，权重越大)
        hardness_term = torch.exp(sim_pp / mining_temp)
        
        # Term 2: False Negative Suppression (状态越像，权重越小)
        # 1 - sim_cc: 状态差异度。
        # 如果 sim_cc 接近 1 (真兄弟)，则权重归零。
        # clamp(min=0) 防止数值误差导致负数
        divergence_term = (1.0 - sim_cc).clamp(min=0.0)
        
        # 组合权重 (detach 很重要，不传导梯度到权重计算本身)
        weights = (hardness_term * divergence_term).detach()
        
        # 排除对角线 (自己不是负样本)
        weights = weights * (1 - mask_diag)
        
        # -----------------------------------------------------------
        # 4. 计算 InfoNCE Loss
        # -----------------------------------------------------------
        
        # 分母：加权后的负样本 + 正样本
        exp_logits = torch.exp(logits)
        
        # sum_{k != i} ( w_{ik} * exp(sim) )
        neg_sum = (weights * exp_logits).sum(dim=1) 
        
        # 分子：正样本
        pos_term = exp_logits.diag()
        
        # Loss
        loss = -torch.log(pos_term / (pos_term + neg_sum + 1e-9)).mean()
        
        return loss

    def compute_hard_negative_loss_mean(self, z_partial, z_complete, temp=0.07, mining_temp=0.1):
        """
        Modified to use Channel-wise Average Similarity.
        
        Args:
            z_partial: [B, C, D] (注意：这里不要在外面 Flatten，保持 C 维度)
            z_complete: [B, C, D]
            temp: InfoNCE 的温度系数
            mining_temp: 控制 Hard Negative 权重的敏感度
        """
        B, C, D = z_partial.shape
        
        # 1. 归一化 (Normalize along the last dimension D)
        # 这样每个通道的 embedding 都是单位向量
        z_p_norm = F.normalize(z_partial, dim=-1) # [B, C, D]
        z_c_norm = F.normalize(z_complete, dim=-1) # [B, C, D]
        
        # ------------------------------------------------------------------
        # 核心修改：使用 einsum 计算 "通道对齐" 的相似度矩阵
        # ------------------------------------------------------------------
        
        # 2. 计算 Sim_Obs (用于挖掘难样本): z_partial vs z_partial
        # 逻辑: 
        # i: batch idx 1, j: batch idx 2, c: channel, d: dimension
        # 'icd, jcd -> ij' 表示：
        # 对每一对样本 (i, j)，计算它们在相同通道 c 上的点积 (d维度求和)，
        # 然后将所有通道 c 的结果相加。
        # 最后除以 C，得到平均余弦相似度。
        
        sim_obs = torch.einsum('icd, jcd -> ij', z_p_norm, z_p_norm) / C  # [B, B]
        
        # 3. 计算 Logits (用于 Loss): z_partial vs z_complete
        logits = torch.einsum('icd, jcd -> ij', z_p_norm, z_c_norm) / C   # [B, B]
        
        # 除以温度系数
        logits = logits / temp

        # ------------------------------------------------------------------
        # 下面的逻辑保持不变 (Hard Negative Weighting)
        # ------------------------------------------------------------------
        
        # 4. 计算动态权重
        mask_diag = torch.eye(B, device=z_partial.device)
        
        # 基于"通道平均相似度"来判断是否像
        weights = torch.exp(sim_obs / mining_temp).detach()
        weights = weights * (1 - mask_diag)
        
        # 5. 计算分母
        exp_logits = torch.exp(logits)
        neg_sum = (weights * exp_logits).sum(dim=1) # [B]
        
        # 6. 计算分子
        pos_term = exp_logits.diag() # [B]
        
        # 7. Final Loss
        loss = -torch.log(pos_term / (pos_term + neg_sum + 1e-9)).mean()
        
        return loss
    

    def forward(self, batch_x, batch_x_full, batch_mask, x_mark=None):
        """
        Training Step:
        batch_x: [B, L, C] (有缺失，缺失处填0)
        batch_x_full: [B, L, C] (完整)
        batch_mask: [B, C] (1代表观测，0代表缺失)
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
        训练对比学习模型
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
        for epoch in range(self.args.encoder_epochs):  # 可调整为 configs.epochs
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
            
            # 验证阶段
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
                    counter = 0  # 重置计数器
                    print(f'--> Best model updated at epoch {epoch+1}!')
                else:
                    counter += 1
                    print(f'--> EarlyStopping counter: {counter} out of {patience}')
                    if counter >= patience:
                        print("Early stopping triggered. Training stopped.")
                        break # 跳出 epoch 循环
            else: 
                best_checkpoints = self.state_dict()
            
            # 学习率调度
            scheduler.step()
        self.load_state_dict(best_checkpoints)
        print("🎉 Training completed!")
        return self

# ================= 使用示例 =================
if __name__ == "__main__":
    # 1. 模拟配置
    class Configs:
        seq_len = 96
        d_model = 128
        embed = 'timeF' # 假设用 Time Feature 编码
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
    
    # 2. 模拟数据 (Batch=32, Length=96, Channels=7)
    B, L, C = 32, 96, 7
    batch_x_full = torch.randn(B, L, C)
    
    # 模拟缺失：随机 Mask 掉部分通道
    batch_mask = torch.randint(0, 2, (B, C)).float() # [B, C], 0 or 1
    # 保证至少有一个观测通道，防止除0
    batch_mask[:, 0] = 1.0 
    
    # 构造 batch_x (缺失处填0)
    batch_x = batch_x_full * batch_mask.unsqueeze(1)
    
    # 3. Forward 计算 Loss
    loss = model(batch_x, batch_x_full, batch_mask)
    print(f"Contrastive Loss: {loss.item()}")

    # 4. 推理/检索阶段 (只用 get_representation)
    model.eval()
    with torch.no_grad():
        # Query Embedding (Partial)
        query_emb = model.get_representation(batch_x, None, batch_mask)
        # Key Embedding (Full)
        key_emb = model.get_representation(batch_x_full, None, None)
        
        # 计算相似度
        query_emb = F.normalize(query_emb, dim=1)
        key_emb = F.normalize(key_emb, dim=1)
        sim = torch.matmul(query_emb, key_emb.T)
        print(f"Similarity Matrix Shape: {sim.shape}") # [32, 32]