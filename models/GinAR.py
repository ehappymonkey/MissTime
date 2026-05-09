import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ginarCell(nn.Module):
    def __init__(self, num_id, in_size, emb_size, grap_size, dropout):
        super(ginarCell, self).__init__()
        self.emb_size = emb_size
        self.num_id = num_id
        self.emb = nn.Conv1d(in_channels=in_size, out_channels=emb_size, kernel_size=1)
        self.emb2 = nn.Linear(num_id, num_id)
        self.att = InterpositionAttention(emb_size, emb_size, num_id, grap_size, dropout)

        self.linear1 = nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1, bias=False)
        self.linear2 = nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1, bias=True)

        self.layernorm = nn.LayerNorm([emb_size, num_id])
        self.dropout = nn.Dropout(dropout)

        ### Adaptive graph
        self.GL = nn.Parameter(torch.FloatTensor(num_id, grap_size))
        nn.init.kaiming_uniform_(self.GL)
        self.GL_linear = nn.Linear(grap_size, emb_size, bias=False)
        # 注意：这里需要确保输入维度和Linear定义匹配，之前分析是匹配的
        self.GL_linear2 = nn.Linear(emb_size * 2, emb_size * 2, bias=False)

    def forward(self, x, ct, graph_data):
        ### embeding and Inductive attention
        # x: [B, emb_size, N]
        x = self.att(self.emb2(F.leaky_relu(self.emb(x))).transpose(-2, -1))

        # Predefined graph
        graph_data1 = graph_data[0].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data1 = ginarCell.calculate_laplacian_with_self_loop(graph_data1)
        graph_data2 = graph_data[1].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data2 = ginarCell.calculate_laplacian_with_self_loop(graph_data2)

        ### Adaptive graph
        B, _, _ = x.shape
        GL_embed = self.GL_linear(self.GL.unsqueeze(0).expand(B, -1, -1))
        # 拼接 x.T [B, N, emb] 和 GL_embed [B, N, emb] -> [B, N, 2*emb]
        GL_embed = self.GL_linear2(torch.cat([x.transpose(-2, -1), GL_embed], dim=-1))
        graph_learn = torch.eye(self.num_id).to(x.device) + F.softmax(F.relu(GL_embed @ GL_embed.transpose(-2, -1)), dim=-1)

        ### GinAR cell
        # 这里的计算没问题
        term1 = self.linear1(x) # Conv1d over N
        term1_do = self.dropout(term1)
        
        # 核心修复 1: 确保 ft 和 rt 使用 Sigmoid 激活函数
        # 原始代码使用 GELU 会导致数值爆炸
        
        # Update Gate
        ft_input = self.layernorm(self.dropout(self.linear2(x)) @ graph_learn + self.linear2(self.dropout(self.linear2(x)) @ graph_data1) @ graph_data2)
        ft = torch.sigmoid(ft_input) # <--- 修改这里：GELU -> Sigmoid

        # Reset Gate
        rt_input = self.layernorm(self.dropout(self.linear2(x)) @ graph_learn + self.linear2(self.dropout(self.linear2(x)) @ graph_data1) @ graph_data2)
        rt = torch.sigmoid(rt_input) # <--- 修改这里：GELU -> Sigmoid

        # Cell Content
        x_new = self.layernorm(term1_do @ graph_learn + self.linear1(term1_do @ graph_data1) @ graph_data2)
        
        # GRU 更新公式: ft * ct + (1-ft) * x_new
        # 只有当 ft 在 [0,1] 之间时，这才是插值，否则就是发散
        ct = ft * ct + x_new - ft * x_new
        ht = rt * F.elu(ct) + x - rt * x
        
        return ht, ct

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix):
        # 核心修复 2: 增加 epsilon 防止除以 0
        row_sum = matrix.sum(1) + 1e-6 
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian


class InterpositionAttention(nn.Module):
    # 这部分代码通常没问题，保持原样即可
    def __init__(self, in_c, out_c, num_id, grap_size, dropout):
        super(InterpositionAttention, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_id = num_id
        self.drop = dropout
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Parameter(torch.FloatTensor(size=(in_c, out_c)))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.FloatTensor(size=(2 * out_c, 1)))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU()
        self.GL = nn.Parameter(torch.FloatTensor(num_id, grap_size))
        nn.init.kaiming_uniform_(self.GL)

        self.GL2 = nn.Parameter(torch.FloatTensor(grap_size, num_id))
        nn.init.kaiming_uniform_(self.GL2)

    def forward(self, inp):
        adj = F.softmax(F.relu(self.GL @ self.GL.transpose(-2, -1)), dim=-1)
        B, N = inp.size(0), inp.size(1)
        adj = adj + torch.eye(N, dtype=adj.dtype, device=adj.device)

        h = torch.matmul(inp, self.W)
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_c)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.dropout(F.softmax(attention, dim=2))
        h_prime = torch.matmul(attention, self.dropout(h))
        h_prime = F.relu(h_prime)
        return h_prime.transpose(-2, -1)


class GinAR(nn.Module):
    def __init__(self, input_len, num_id, out_len, in_size, emb_size, grap_size, layer_num, dropout, adj_mx=None):
        super(GinAR, self).__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.num_id = num_id
        self.layer_num = layer_num
        self.emb_size = emb_size
        self.graph_data = adj_mx

        self.ginar_first = ginarCell(num_id, in_size, emb_size, grap_size, dropout)
        self.ginar_other = ginarCell(num_id, emb_size, emb_size, grap_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.lay_norm = nn.LayerNorm([input_len, num_id])

        self.decoder = nn.Conv2d(in_channels=layer_num, out_channels=out_len, kernel_size=(1, emb_size))
        self.output = nn.Conv2d(in_channels=out_len, out_channels=out_len, kernel_size=1)

    def forward(self, history_data):
        # 核心修复 3: 处理输入中的 NaN (防止脏数据污染)
        # 如果你的数据是 Imputation 任务，输入可能包含 NaN，Conv1d 会直接输出 NaN
        history_data = torch.nan_to_num(history_data, nan=0.0)
        
        x = history_data.unsqueeze(1) # [B, 1, L, N]
        B, C, L, N = x.shape

        if self.graph_data is None:
            zero_adj = torch.zeros(N, N, device=x.device)
            graph_data = [zero_adj, zero_adj]
        else:
            graph_data = self.graph_data

        final_result = None
        for z in range(self.layer_num):
            result = None
            ct = torch.zeros(B, self.emb_size, N).to(x.device)
            if z == 0:
                for j in range(self.input_len):
                    ht, ct = self.ginar_first(x[:, :, j, :], ct, graph_data)
                    if j == 0:
                        result = ht.unsqueeze(-2)
                    else:
                        result = torch.cat([result, ht.unsqueeze(-2)], dim=-2)
            else:
                for j in range(self.input_len):
                    ht, ct = self.ginar_other(x[:, :, j, :], ct, graph_data)
                    if j == 0:
                        result = ht.unsqueeze(-2)
                    else:
                        result = torch.cat([result, ht.unsqueeze(-2)], dim=-2)

            x = result.clone()
            last_step_result = result[:, :, -1, :]

            if z == 0:
                final_result = last_step_result.transpose(-2, -1).unsqueeze(1)
            else:
                final_result = torch.cat([final_result, last_step_result.transpose(-2, -1).unsqueeze(1)], dim=1)

        x = self.dropout(self.decoder(final_result))
        x = self.output(x)
        return x.squeeze(-1)
    
if __name__ == '__main__':
    model = GinAR(input_len=96, num_id=7, out_len=96, in_size=1, 
              emb_size=32, grap_size=8, layer_num=2, dropout=0.1, 
              adj_mx=None) # <--- 这里

    input_data = torch.randn(32, 96, 7)
    output = model(input_data)
    print(output)