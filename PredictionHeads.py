import torch
import torch.nn as nn
import torch.nn.functional as F

# two pred-heads

class End2EndPredictionHead(nn.Module):
    def __init__(self, dim):
        super(End2EndPredictionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output two values
            nn.Sigmoid()
        )
    
    def forward(self, vq_out):
        """
        Forward pass for End2EndPredictionHead.
        
        Args:
            vq_out (torch.Tensor): Input tensor of shape [B, dim]
        
        Returns:
            torch.Tensor: Clamped output tensor of shape [B, 2] 
        """
        z = self.mlp(vq_out)  # Pass through MLP, shape [B, 2]
        # Compute a and b based on z[:, 0] and z[:, 1]
        output = torch.stack([z[:, 0] - z[:, 1] / 2, z[:, 0] + z[:, 1] / 2], dim=1)
        return torch.clamp(output, 0.0, 1.0)  # Ensure values are within [0, 1]

class FramewisePredictionHead(nn.Module):
    def __init__(self, dim):
        super(FramewisePredictionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
    def forward(self, sr_out):
        """
        Forward pass for FramewisePredictionHead.
        
        Args:
            sr_out (torch.Tensor): Enhanced scanning reference, shape [B, Tr, dim]
        
        Returns:
            torch.Tensor: Predicted sequence with shape [B, Tr, 4]
        """
        out = self.mlp(sr_out)  # [B, Tr, 4]
        return F.softmax(out, dim=-1)

