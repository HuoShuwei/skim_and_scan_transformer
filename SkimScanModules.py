import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from BaseFunctions import PositionalEncoding, AttentionalMixer, MLP 

class AttentionalCorrelation(nn.Module):
    def __init__(self, dim, num_heads, max_len=5000):
        super(AttentionalCorrelation, self).__init__()
        self.pos_encoding_q = PositionalEncoding(dim, max_len)
        self.pos_encoding_kv = PositionalEncoding(dim, max_len)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, q_in, kv_in):
        """
        Forward pass for AttentionalCorrelation.
        
        Args:
            q_in (torch.Tensor): Query tensor of shape [B, N, C]
            kv_in (torch.Tensor): Key/Value tensor of shape [B, M, C]
        
        Returns:
            torch.Tensor: Output tensor after cross-attention and projection, shape [B, N, C]
        """
        q_in = self.pos_encoding_q(q_in)      # Add positional encoding to query
        kv_in = self.pos_encoding_kv(kv_in)  # Add positional encoding to key and value
        
        B, N, C = q_in.shape
        _, M, _ = kv_in.shape
        
        q = self.q(q_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(kv_in).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(kv_in).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class AttentionBasedTransformUnit(nn.Module):
    def __init__(self, dim, num_heads, max_len=5000):
        super(AttentionBasedTransformUnit, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionalMixer(dim, num_heads, max_len)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)
        
    def forward(self, x):
        """
        Forward pass for AttentionBasedTransformUnit.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C]
        
        Returns:
            torch.Tensor: Output tensor of shape [B, N, C]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SkimmingBranch(nn.Module):
    def __init__(self, dim, num_heads, max_len=5000):
        super(SkimmingBranch, self).__init__()
        self.attn_mix = AttentionalMixer(dim, num_heads, max_len)
        self.mlp1 = MLP(dim, dim * 4)
        self.mlp2 = MLP(dim, dim * 4)
        
    def forward(self, vq, sq):
        """
        Forward pass for SkimmingBranch.
        
        Args:
            vq (torch.Tensor): Vector query tensor of shape [B, dim]
            sq (torch.Tensor): Sequence query tensor of shape [B, Tq, dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - vq_star: Transformed vector query, shape [B, dim]
                - sq_star: Transformed sequence query, shape [B, Tq, dim]
        """
        # Convert vq from [B, dim] to [B, 1, dim]
        vq = vq.unsqueeze(1)  # [B, 1, dim]
        combined = torch.cat([vq, sq], dim=1)  # [B, Tq+1, dim]
        mixed = self.attn_mix(combined)
        
        vq_star = mixed[:, 0:1, :]  # [B, 1, dim]
        sq_star = mixed[:, 1:, :]   # [B, Tq, dim]
        
        vq_star = self.mlp2(self.mlp1(vq_star))  # [B, 1, dim]
        return vq_star.squeeze(1), sq_star  # Returns [B, dim] and [B, Tq, dim]

class ScanningBranch(nn.Module):
    def __init__(self, dim, num_heads, max_len=5000):
        super(ScanningBranch, self).__init__()
        self.attn_mix = AttentionalMixer(dim, num_heads, max_len)
        self.mlp1 = MLP(dim, dim * 4)
        self.mlp2 = MLP(dim, dim * 4)
        self.cross_attn = AttentionalCorrelation(dim, num_heads, max_len)
        self.final_mlp = MLP(dim, dim * 4)
        
    def forward(self, sr, sq_star):
        """
        Forward pass for ScanningBranch.
        
        Args:
            sr (torch.Tensor): Scanning reference tensor of shape [B, Tr, dim]
            sq_star (torch.Tensor): Transformed sequence query, shape [B, Tq, dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - sr_star: Transformed scanning reference, shape [B, Tr, dim]
                - sr_plus: Enhanced scanning reference, shape [B, Tr, dim]
        """
        sr_mixed = self.attn_mix(sr)
        sr_star = self.mlp2(self.mlp1(sr_mixed))
        sr_plus = self.final_mlp(self.cross_attn(sr_star, sq_star))
        return sr_star, sr_plus

class SubModule(nn.Module):
    def __init__(self, dim, num_heads, max_len=5000):
        super(SubModule, self).__init__()
        self.skim = SkimmingBranch(dim, num_heads, max_len)
        self.scan = ScanningBranch(dim, num_heads, max_len)
        self.final_mlp = MLP(dim, dim * 4)
        
    def convolutional_correlation(self, vq_star, sr_star):
        """
        Performs convolutional correlation between vq_star and sr_star using conv1d.
        
        Args:
            vq_star (torch.Tensor): Transformed vector query, shape [B, dim]
            sr_star (torch.Tensor): Transformed scanning reference, shape [B, N, dim]
        
        Returns:
            torch.Tensor: Enhanced vector query, shape [B, dim]
        """
        B, C = vq_star.shape
        N = sr_star.shape[1]
        
        # Reshape sr_star for convolution: [B, dim, N]
        sr_reshaped = sr_star.permute(0, 2, 1)  # [B, dim, N]
        
        # Reshape vq_star as convolution weights: [B, dim, 1]
        vq_weights = vq_star.unsqueeze(-1)  # [B, dim, 1]
        
        # Merge batch and channel dimensions for grouped convolution
        sr_reshaped = sr_reshaped.reshape(1, B * C, N)      # [1, B*dim, N]
        vq_weights = vq_weights.reshape(B * C, 1, 1)      # [B*dim, 1, 1]
        
        # Perform conv1d with groups=B*dim (each group has 1 input and 1 output channel)
        weights = F.conv1d(sr_reshaped, vq_weights, groups=B * C)  # [1, B*dim, N]
        
        # Reshape weights back to [B, dim, N]
        weights = weights.view(B, C, N)  # [B, dim, N]
        
        # Aggregate weights across channels (e.g., average)
        weights = weights.mean(dim=1)  # [B, N]
        
        # Apply softmax to obtain attention weights
        weights = F.softmax(weights, dim=-1)  # [B, N]
        
        # Compute weighted sum: [B, dim]
        vq_plus = torch.bmm(weights.unsqueeze(1), sr_star).squeeze(1)  # [B, dim]
        
        # Pass through final MLP
        return self.final_mlp(vq_plus)  # [B, dim]
        
    def forward(self, vq, sq, sr):
        """
        Forward pass for SubModule.
        
        Args:
            vq (torch.Tensor): Vector query tensor, shape [B, dim]
            sq (torch.Tensor): Sequence query tensor, shape [B, Tq, dim]
            sr (torch.Tensor): Scanning reference tensor, shape [B, Tr, dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - vq_plus: Enhanced vector query, shape [B, dim]
                - sq_star: Transformed sequence query, shape [B, Tq, dim]
                - sr_plus: Enhanced scanning reference, shape [B, Tr, dim]
        """
        vq_star, sq_star = self.skim(vq, sq)
        sr_star, sr_plus = self.scan(sr, sq_star)
        vq_plus = self.convolutional_correlation(vq_star, sr_star)
        return vq_plus, sq_star, sr_plus
    

class SkimAndScanModule(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, max_len=5000):
        super(SkimAndScanModule, self).__init__()
        self.blocks = nn.ModuleList([
            SubModule(dim, num_heads, max_len) for _ in range(num_blocks)
        ])
        
    def forward(self, vq, sq, sr):
        """
        Forward pass for SkimAndScanModule.
        
        Args:
            vq (torch.Tensor): Vector query tensor, shape [B, dim]
            sq (torch.Tensor): Sequence query tensor, shape [B, Tq, dim]
            sr (torch.Tensor): Scanning reference tensor, shape [B, Tr, dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - vq: Enhanced vector query, shape [B, dim]
                - sq: Transformed sequence query, shape [B, Tq, dim]
                - sr: Enhanced scanning reference, shape [B, Tr, dim]
        """
        for block in self.blocks:
            vq, sq, sr = block(vq, sq, sr)
        return vq, sq, sr