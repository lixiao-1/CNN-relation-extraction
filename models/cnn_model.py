import torch
import torch.nn as nn

class RE_CNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=300, filter_sizes=[3, 4, 5], num_filters=128, dropout_rate=0.1):
        super(RE_CNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_filters, (fs, embedding_dim)),
                nn.BatchNorm2d(num_filters),
                nn.ReLU()
            ) for fs in filter_sizes
        ])
        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * num_filters, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.unsqueeze(1)
        conv_outputs = []
        for conv in self.convs:
            x = conv(embedded)
            x = nn.functional.max_pool1d(x.squeeze(3), x.size(2)).squeeze(2)
            conv_outputs.append(x)
        x = torch.cat(conv_outputs, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits