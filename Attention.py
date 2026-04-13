import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16, dropout_rate=0.1):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        beta = self.dropout(beta)  # Add Dropout
        return (beta * z).sum(1), beta