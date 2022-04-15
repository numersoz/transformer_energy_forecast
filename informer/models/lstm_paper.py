import torch
import torch.nn as nn
import torch.nn.functional as F

class DTO_DRNN(nn.Module):
    def __init__(self, feature_count=6, lstm_hidden_size=400, prediction_length=336, dropout=0.1, device=torch.device('cuda:0'), dense_sizes=[15, 15], *args, **kwargs):
        super(DTO_DRNN, self).__init__()
        self.lstm = nn.LSTM(feature_count, lstm_hidden_size, batch_first = True, num_layers=2)
        self.dense_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, dense_sizes[0]),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(dense_sizes[0], dense_sizes[1]),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(dense_sizes[1], prediction_length)
        )
        self.init_hidden()
    
    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, in_mat, *args, **kwargs):
        rec, _ = self.lstm(in_mat)
        return torch.unsqueeze(self.dense_block(rec)[:, -1, :], 2)
