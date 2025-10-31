import torch.nn as nn


class Embedding(nn.Module):
    """Simple embedding layer for node features."""
    
    def __init__(self, in_nf: int, out_nf: int):
        super().__init__()
        self.emb = nn.Embedding(in_nf, out_nf)
    
    def forward(self, x):
        return self.emb(x)
