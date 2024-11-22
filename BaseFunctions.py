import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough positional encodings
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer('pe', pe)  # Not trained

    def forward(self, x):
        """
        Adds positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C]
        
        Returns:
            torch.Tensor: Output tensor with positional encoding added, shape [B, N, C]
        """
        B, N, C = x.size()
        if N > self.pe.size(1):
            raise ValueError(f"Sequence length {N} exceeds maximum length {self.pe.size(1)}")
        pe = self.pe[:, :N, :]  # [1, N, C]
        return x + pe  # Broadcast addition

class AttentionalMixer(nn.Module):
    def __init__(self, dim, num_heads, max_len=5000):
        super(AttentionalMixer, self).__init__()
        self.pos_encoding = PositionalEncoding(dim, max_len)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Forward pass for AttentionalMixer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C]
        
        Returns:
            torch.Tensor: Output tensor after attention and projection, shape [B, N, C]
        """
        x = self.pos_encoding(x)  # Add positional encoding
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        """
        Forward pass for MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape [*, dim]
        
        Returns:
            torch.Tensor: Output tensor of shape [*, dim]
        """
        return self.fc2(self.act(self.fc1(x)))