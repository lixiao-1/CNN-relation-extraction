import torch
import torch.nn as nn

class RE_CNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, filter_sizes=[3, 4, 5], num_filters=100):
        super(RE_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.unsqueeze(1)
        conv_outputs = []
        for conv in self.convs:
            x = conv(embedded)
            x = nn.functional.relu(x).squeeze(3)
            x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
            conv_outputs.append(x)
        x = torch.cat(conv_outputs, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits