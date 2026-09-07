import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==========================================
# 核心组件: Set Transformer Blocks (MAB, SAB, PMA)
# ==========================================

class MAB(nn.Module):
    """
    Multihead Attention Block (MAB)
    Structure: Attention -> Add & Norm -> rFF -> Add & Norm
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        
        # Linear Projections
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        
        # Output Projection
        self.fc_o = nn.Linear(dim_V, dim_V)
        
        # Layer Normalization
        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        
        # Row-wise Feed Forward (rFF)
        # Standard FFN in Transformer: Linear -> ReLU -> Linear
        self.rff = nn.Sequential(
            nn.Linear(dim_V, dim_V * 4), # Usually expansion factor is 4
            nn.ReLU(),
            nn.Linear(dim_V * 4, dim_V)
        )

    def forward(self, Q, K):
        # Q: [Batch, N_q, Dim_Q]
        # K: [Batch, N_k, Dim_K] (Assume V is same source as K)
        
        # 1. Project Q, K, V
        Q_proj = self.fc_q(Q) # [B, N_q, D]
        K_proj = self.fc_k(K) # [B, N_k, D]
        V_proj = self.fc_v(K) # [B, N_k, D]
        
        dim_split = self.dim_V // self.num_heads
        
        # 2. Split Heads: [B, N, H, D_head] -> [B*H, N, D_head]
        Q_ = torch.cat(Q_proj.split(dim_split, 2), 0)
        K_ = torch.cat(K_proj.split(dim_split, 2), 0)
        V_ = torch.cat(V_proj.split(dim_split, 2), 0)
        
        # 3. Scaled Dot-Product Attention
        # A: [B*H, N_q, N_k]
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        
        # 4. Weighted Sum: [B*H, N_q, D_head]
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        
        # 5. Output Projection + Residual + Norm
        O = O + self.fc_o(O)
        O = self.ln0(O)
        
        # 6. Feed Forward + Residual + Norm
        O = O + self.rff(O)
        O = self.ln1(O)
        
        return O

class SAB(nn.Module):
    """
    Set Attention Block (SAB) = MAB(X, X)
    Implements Self-Attention for the set.
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=True):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA)
    Aggregates set using learnable seed vectors.
    """
    def __init__(self, dim, num_heads, num_seeds, ln=True):
        super(PMA, self).__init__()
        # S: [1, num_seeds, dim] - Learnable Seed Vectors
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        
        
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        # Broadcast S to match Batch size of X
        # X: [Batch, N, Dim]
        # S: [Batch, num_seeds, Dim]
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class RoleEncoder(nn.Module):
    def __init__(self, input_dim, role_embed_dim, n_heads=2):
        super(RoleEncoder, self).__init__()
        # Input Projection: 线性映射
        self.input_proj = nn.Linear(input_dim, role_embed_dim)
        # SAB Blocks: 负责交互和非线性变换
        self.sab1 = SAB(role_embed_dim, role_embed_dim, n_heads, ln=True)
        self.sab2 = SAB(role_embed_dim, role_embed_dim, n_heads, ln=True)
        
    def forward(self, inputs):
        # inputs: [B*T, N, Input_Dim]
        x = self.input_proj(inputs) 
        x = self.sab1(x)
        roles = self.sab2(x) # [B*T, N, Role_Dim]
        return roles



# ==========================================
# 主体类: QSTransMixer
# ==========================================

class QSTransMixer_roleEncoder(nn.Module):
    def __init__(self, args):
        super(QSTransMixer, self).__init__()
        
        self.args = args
        self.n_agents = args.n_agents
        
        # --- Dimensions ---
        # 状态维度 (e.g., 322)
        self.state_dim = int(np.prod(args.state_shape)) 
        # 观测维度 (e.g., 176) - 假设是打平后的 obs
        self.obs_dim = args.obs_shape 
        # 嵌入维度 d_model (e.g., 64) - 对应 yaml 中的 mixing_embed_dim
        self.embed_dim = args.mixing_embed_dim
        # 增加role embed
        self.role_embed_dim = args.role_embed_dim
        
        # --- Hyperparameters ---
        self.n_heads = getattr(args, "n_head", 4) # Attention heads
        self.n_pma_buffer = getattr(args, "pma_buffer_size", 4) # Buffer size m=4
        
        #role encoder
        self.role_input_dim = 1 + self.obs_dim + self.state_dim  # Q_i + Obs_i + State
        self.role_encoder = RoleEncoder(self.role_input_dim, self.role_embed_dim, self.n_heads)

        ## main mixer
        # --- 1. Input Calculation ---
        # Input Composition: [Q_i (1) + Obs_i (obs_dim) + State (state_dim) + role_embedding (role_embed_dim)]
        self.input_total_dim = 1 + self.obs_dim + self.state_dim + self.role_embed_dim
        
        # --- 2. Projection Layer ---
        # Maps raw huge input to embed_dim (509 -> 64)
        self.input_proj = nn.Linear(self.input_total_dim, self.embed_dim)
        
        # --- 3. Encoder (2 layers of SAB) ---
        self.encoder = nn.Sequential(
            SAB(self.embed_dim, self.embed_dim, self.n_heads, ln=True),
            SAB(self.embed_dim, self.embed_dim, self.n_heads, ln=True)
        )
        
        # --- 4. Decoder (2 layers of PMA) ---
        # Layer 1: Compress N agents -> m=4 features
        self.decoder_pma1 = PMA(self.embed_dim, self.n_heads, self.n_pma_buffer, ln=True)
        # Layer 2: Compress m=4 features -> k=1 embedding
        self.decoder_pma2 = PMA(self.embed_dim, self.n_heads, 1, ln=True)
        
        # --- 5. Output Head ---
        # Final MLP to produce Q_tot
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states, obs):
        """
        Args:
            agent_qs: [Batch, Time, N_Agents]
            states:   [Batch, Time, State_Dim]
            obs:      [Batch, Time, N_Agents, Obs_Dim]
        Returns:
            q_tot:    [Batch, Time, 1]
        """
        bs, t, n_agents = agent_qs.size()
        
        # -----------------------------------------------
        # Step 1: Data Reshaping & Input Construction
        # -----------------------------------------------
        # Flatten Batch and Time for Transformer processing: [B*T, ...]
        batch_size = bs * t
        
        # 1. Process States: [B, T, State_Dim] -> [B*T, N, State_Dim]
        # Repeat global state for each agent
        states_reshaped = states.reshape(batch_size, 1, self.state_dim)
        states_expanded = states_reshaped.repeat(1, n_agents, 1) 
        
        # 2. Process Q-Values: [B, T, N] -> [B*T, N, 1]
        qvals_reshaped = agent_qs.reshape(batch_size, n_agents, 1)
        
        # 3. Process Obs: [B, T, N, Obs_Dim] -> [B*T, N, Obs_Dim]
        obs_reshaped = obs.reshape(batch_size, n_agents, self.obs_dim)
        
        # 4. Generate Agent IDs: [B*T, N, N] (One-hot)
        # Create identity matrix [N, N] and repeat for batch
        # agent_ids = torch.eye(n_agents, device=agent_qs.device).unsqueeze(0)
        # agent_ids_expanded = agent_ids.repeat(batch_size, 1, 1)

        #4. get role embedding
        role_inputs = torch.cat([qvals_reshaped, obs_reshaped, states_expanded], dim=2)
        role_embed = self.role_encoder(role_inputs)
        
        # 5. Concatenate All: [B*T, N, Total_Dim]
        # Order: Q (1) + Obs (176) + State (322) + role_embed (32) = 531
        inputs = torch.cat([qvals_reshaped, obs_reshaped, states_expanded, role_embed], dim=2)
        
        # -----------------------------------------------
        # Step 2: Transformer Pipeline
        # -----------------------------------------------
        
        # 1. Projection: [B*T, N, 509] -> [B*T, N, 64]
        x = self.input_proj(inputs)
        
        # 2. Encoder (SABs): [B*T, N, 64] -> [B*T, N, 64]
        # Agents interact with each other (Self-Attention)
        h_enc = self.encoder(x)
        
        # 3. Decoder PMA 1 (Buffer): [B*T, N, 64] -> [B*T, 4, 64]
        # Aggregates info into 4 latent features
        h_dec1 = self.decoder_pma1(h_enc)
        
        # 4. Decoder PMA 2 (Aggregation): [B*T, 4, 64] -> [B*T, 1, 64]
        # Aggregates info into 1 final embedding
        h_dec2 = self.decoder_pma2(h_dec1)
        
        # -----------------------------------------------
        # Step 3: Final Output
        # -----------------------------------------------
        
        # [B*T, 1, 64] -> [B*T, 1, 1]
        q_tot_flat = self.output_head(h_dec2)
        
        # Reshape back to [Batch, Time, 1]
        q_tot = q_tot_flat.view(bs, t, 1)
        
        return q_tot