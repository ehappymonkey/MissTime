
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 假设以下组件已在环境中定义，或者你需要从原来的DUET文件中import
# from layers.Embed import Linear_extractor_cluster, Mahalanobis_mask
# from layers.Transformer_EncDec import Encoder, EncoderLayer, AttentionLayer, FullAttention

# from models.duet_utils import Linear_extractor_cluster


import torch.nn as nn
import torch
from math import sqrt

import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import math
import torch.fft
from einops import rearrange
from models.duet_utils import encoder, RevIN, Normal, SparseDispatcher
from models.duet_utils import Linear_extractor as expert


class Linear_extractor_cluster(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, config):
        super(Linear_extractor_cluster, self).__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.input_size = config.seq_len
        self.k = config.k
        # instantiate experts
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])
        self.W_h = nn.Parameter(torch.eye(self.num_experts))
        self.gate = encoder(config)
        self.noise = encoder(config)

        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)

        self.CI = config.CI
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = self.gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (
            top_k_logits.sum(1, keepdim=True) + 1e-6
        )  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        if self.CI:
            x_norm = rearrange(x, "(x y) l c -> x l (y c)", y=self.n_vars)
            x_norm = self.revin(x_norm, "norm")
            x_norm = rearrange(x_norm, "x l (y c) -> (x y) l c", y=self.n_vars)
        else:
            x_norm = self.revin(x, "norm")

        expert_inputs = dispatcher.dispatch(x_norm)

        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)

        return y, loss
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # if self.mask_flag:
        #     large_negative = -math.log(1e10)
        #     attention_mask = torch.where(attn_mask == 0, torch.tensor(large_negative), attn_mask)
        #
        #     scores = scores * attention_mask
        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)

            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size):
        super(Mahalanobis_mask, self).__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(
            torch.randn(frequency_size, frequency_size), requires_grad=True
        )

    def calculate_prob_distance(self, X):
        XF = torch.abs(torch.fft.rfft(X, dim=-1))
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)

        # B x C x C x D
        diff = X1 - X2

        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)

        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)

        # exp_dist = torch.exp(-dist)
        exp_dist = 1 / (dist + 1e-10)
        # 对角线置零

        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()

        # B x C x C
        p = exp_dist / exp_max

        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)

        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99

        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, "b c d -> (b c d) 1")
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(
            resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d
        )
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)

        # bernoulli中两个通道有关系的概率
        sample = self.bernoulli_gumbel_rsample(p)

        mask = sample.unsqueeze(1)
        cnt = torch.sum(mask, dim=-1)
        return mask


                     
class Model(nn.Module):
    """
    DUET Model adapted for Multi-task (Time Series Library Style)
    Original Paper: DUET (Dependence-Unifying Embedding for Transformer)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        # self.CI = configs.CI
        self.CI = True
        configs.CI = True
        configs.noisy_gating = True
        configs.num_experts = 4
        configs.k = 1
        configs.hidden_size = 256
        configs.fc_dropout = 0.2
        
        # --- DUET Core Components ---
        self.cluster = Linear_extractor_cluster(configs)
        self.mask_generator = Mahalanobis_mask(configs.seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # ----------------------------

        # --- Task-specific Heads ---
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.linear_head = nn.Sequential(
                nn.Linear(configs.d_model, configs.pred_len), 
                nn.Dropout(configs.fc_dropout)
            )
            
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # Imputation and Anomaly Detection reconstruct the original sequence (seq_len)
            self.linear_head = nn.Sequential(
                nn.Linear(configs.d_model, configs.seq_len), 
                nn.Dropout(configs.fc_dropout)
            )
            
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # Classification flattens features: (batch, n_vars * d_model) -> num_classes
            self.linear_head = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def get_features(self, input):
        """
        Extracts features using DUET's clustering and masking mechanism.
        Returns: 
            channel_group_feature: [B, N, d_model]
            L_importance: Importance weights from the cluster module
        """
        # x: [batch_size, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(input, "b l n -> (b n) l 1")
            # Note: The 'cluster' module likely handles RevIN normalization internally
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            
            temporal_feature = rearrange(
                reshaped_output, "(b n) l 1 -> b l n", b=input.shape[0]
            )
        else:
            temporal_feature, L_importance = self.cluster(input)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")
        
        if self.enc_in > 1:
            changed_input = rearrange(input, "b l n -> b n l")
            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, attention = self.Channel_transformer(
                x=temporal_feature, attn_mask=channel_mask
            )
        else:
            channel_group_feature = temporal_feature

        return channel_group_feature, L_importance

    def forecast(self, x_enc):
        # 1. Extract Features
        # channel_group_feature: [B, N, d_model]
        channel_group_feature, L_importance = self.get_features(x_enc)

        # 2. Projection
        # [B, N, d_model] -> [B, N, pred_len]
        output = self.linear_head(channel_group_feature)

        # 3. Denormalization & Formatting
        # [B, N, pred_len] -> [B, pred_len, N]
        output = rearrange(output, "b n d -> b d n")
        
        # Use DUET's internal RevIN for denormalization
        output = self.cluster.revin(output, "denorm")
        
        return output, L_importance

    def imputation(self, x_enc):
        # 1. Extract Features
        channel_group_feature, L_importance = self.get_features(x_enc)

        # 2. Projection (Reconstruction)
        # [B, N, d_model] -> [B, N, seq_len]
        output = self.linear_head(channel_group_feature)

        # 3. Denormalization & Formatting
        # [B, N, seq_len] -> [B, seq_len, N]
        output = rearrange(output, "b n d -> b d n")
        
        # Use DUET's internal RevIN for denormalization
        output = self.cluster.revin(output, "denorm")
        
        return output, L_importance

    def anomaly_detection(self, x_enc):
        # Similar to imputation, we reconstruct the input
        channel_group_feature, L_importance = self.get_features(x_enc)
        
        # [B, N, d_model] -> [B, N, seq_len]
        output = self.linear_head(channel_group_feature)
        
        # [B, N, seq_len] -> [B, seq_len, N]
        output = rearrange(output, "b n d -> b d n")
        
        # Denormalize
        output = self.cluster.revin(output, "denorm")
        
        return output, L_importance

    def classification(self, x_enc):
        # 1. Extract Features
        channel_group_feature, L_importance = self.get_features(x_enc)
        
        # 2. Processing for Classification
        # channel_group_feature: [B, N, d_model]
        # We assume classification doesn't need RevIN denormalization on the logits
        
        # Flatten: [B, N * d_model]
        output = channel_group_feature.reshape(channel_group_feature.shape[0], -1)
        
        output = self.dropout(output)
        output = self.linear_head(output) # [B, num_classes]
        
        return output, L_importance

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # DUET usually doesn't use x_mark (time features) or decoder inputs explicitly 
        # in the provided snippet, so we primarily use x_enc.
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, L_imp = self.forecast(x_enc)
            # TSL Standard: returns (output) or (output, aux_loss)
            # Here we return the tuple as requested to keep L_importance
            return dec_out, L_imp 
            
        if self.task_name == 'imputation':
            dec_out, L_imp = self.imputation(x_enc)
            return dec_out, L_imp
            
        if self.task_name == 'anomaly_detection':
            dec_out, L_imp = self.anomaly_detection(x_enc)
            return dec_out, L_imp
            
        if self.task_name == 'classification':
            dec_out, L_imp = self.classification(x_enc)
            return dec_out, L_imp
            
        return None