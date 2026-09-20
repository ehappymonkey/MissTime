
import torch

def apply_random_channel_mask(x, mask_ratio=0.2):
    """
    x: [Batch, Seq_Len, Channels]
    将随机 mask_ratio 比例的通道全部置为 0
    """
    B, S, C = x.shape
    device = x.device
    
    # 创建一个全 1 的 mask
    mask = torch.ones(B, C).to(device)
    
    # 计算需要丢弃的通道数量
    n_drop = int(C * mask_ratio)
    
    if n_drop > 0:
        for i in range(B):
            # 随机选择要丢弃的通道索引
            drop_indices = torch.randperm(C)[:n_drop]
            mask[i, drop_indices] = 0.0
            
    # 扩展维度以匹配输入 [B, 1, C] ->广播-> [B, S, C]
    # 假设是整条序列的某个变量彻底丢失 (Channel-wise missing)
    mask = mask.unsqueeze(1).expand_as(x)
    
    # 应用 Mask
    masked_x = x * mask
    
    return masked_x, mask
