import torch.nn as nn
import torch.nn.functional as F


class End2EndPredictionHead(nn.Module):
    def __init__(self, dim):
        super(End2EndPredictionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        
    def forward(self, vq_out):
        """
        Forward pass for End2EndPredictionHead.
        
        Args:
            vq_out (torch.Tensor): Enhanced vector query, shape [B, dim]
        
        Returns:
            torch.Tensor: Predicted 2-dimensional vector, shape [B, 2]
        """
        return self.mlp(vq_out)

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

