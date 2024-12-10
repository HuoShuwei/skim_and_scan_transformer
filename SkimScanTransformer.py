# -*- coding: utf-8 -*-
"""
Main file for Skim-and-Scan Transformer
Aug, 2024
@author: Shuwei Huo
"""

import torch
import torch.nn as nn
from SkimScanModules import SkimAndScanModule, AttentionBasedTransformUnit
from PredictionHeads import End2EndPredictionHead, FramewisePredictionHead

class SkimAndScanTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, max_len=5000):
        super(SkimAndScanTransformer, self).__init__()
        self.query_unit = AttentionBasedTransformUnit(dim, num_heads, max_len)
        self.ref_unit = AttentionBasedTransformUnit(dim, num_heads, max_len)
        self.skim_scan = SkimAndScanModule(dim, num_heads, num_blocks, max_len)
        self.end2end_head = End2EndPredictionHead(dim)
        self.framewise_head = FramewisePredictionHead(dim)
        
    def forward(self, query_seq, ref_seq, vq):
        """
        Forward pass for SkimAndScanTransformer.
        
        Args:
            query_seq (torch.Tensor): Query sequence, shape [B, Tq, dim]
            ref_seq (torch.Tensor): Reference sequence, shape [B, Tr, dim]
            vq (torch.Tensor): Vector query, shape [B, dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - vq_out: Enhanced vector query, shape [B, dim]
                - sq_out: Transformed sequence query, shape [B, Tq, dim]
                - sr_out: Enhanced scanning reference, shape [B, Tr, dim]
                - end2end_pred: End-to-End predictions, shape [B, 2]
                - framewise_pred: Framewise predictions, shape [B, Tr, 4]
        """
        # Only transform query_seq and ref_seq
        query_transformed = self.query_unit(query_seq)
        ref_transformed = self.ref_unit(ref_seq)
        
        # vq is directly input to skim_scan
        vq_out, sq_out, sr_out = self.skim_scan(vq, query_transformed, ref_transformed)
        
        # Generate predictions
        end2end_pred = self.end2end_head(vq_out)      # [B, 2]
        framewise_pred = self.framewise_head(sr_out)  # [B, Tr, 4]
        
        return vq_out, sq_out, sr_out, end2end_pred, framewise_pred


if __name__ == '__main__':
    # Example usage
    dim = 512          # embedding dimension
    num_heads = 8      # number of attention heads
    num_blocks = 2     # number of SubModules in SkimAndScanModule
    Tr = 200           # reference sequence length for FramewisePredictionHead
    max_len = 2048     # maximum sequence length for positional encoding

    model = SkimAndScanTransformer(dim, num_heads, num_blocks, max_len)
    
    # Example input dimensions
    batch_size = 16
    Tq = 100  # query sequence length
    
    query_seq = torch.randn(batch_size, Tq, dim)    # [B, Tq, dim]
    ref_seq = torch.randn(batch_size, Tr, dim)      # [B, Tr, dim]
    vq = torch.randn(batch_size, dim)               # [B, dim]
    
    vq_out, sq_out, sr_out, end2end_pred, framewise_pred = model(query_seq, ref_seq, vq)
    
    print("vq_out shape:", vq_out.shape)                # Expected: [16, 512]
    print("sq_out shape:", sq_out.shape)                # Expected: [16, 100, 512]
    print("sr_out shape:", sr_out.shape)                # Expected: [16, 200, 512]
    print("End2EndPredictionHead output shape:", end2end_pred.shape)    # Expected: [16, 2]
    print("FramewisePredictionHead output shape:", framewise_pred.shape)  # Expected: [16, 200, 4]
