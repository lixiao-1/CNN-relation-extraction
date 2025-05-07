import torch
import torch.nn as nn

class RE_CNN(nn.Module):
    def __init__(self,
                 vocab_size=21128,
                 embed_dim=200,
                 pos_dim=50,
                 num_filters=512,
                 kernel_sizes=[3, 5, 7],
                 num_classes=11,
                 num_relations=4):
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 位置嵌入层
        self.pos_embed = nn.Embedding(512, pos_dim)  # 相对位置范围-255~255

        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim + 2 * pos_dim,
                out_channels=num_filters,
                kernel_size=k
            ) for k in kernel_sizes
        ])

        # 实体分类层
        self.entity_classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        # 关系分类层
        self.relation_classifier = nn.Linear(num_filters * len(kernel_sizes), num_relations)

    def forward(self, input_ids, e1_pos, e2_pos):
        batch_size = input_ids.size(0)

        # 词向量
        word_emb = self.embedding(input_ids)  # [B, L, E]

        # 调整 e1_pos 和 e2_pos 的形状
        e1_pos = e1_pos.squeeze(1)
        e2_pos = e2_pos.squeeze(1)

        # 位置特征
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        rel_pos1 = positions - e1_pos.unsqueeze(-1) + 256  # 转换为正索引
        rel_pos2 = positions - e2_pos.unsqueeze(-1) + 256
        pos_emb1 = self.pos_embed(rel_pos1)  # [B, L, P]
        pos_emb2 = self.pos_embed(rel_pos2)  # [B, L, P]

        # 特征拼接
        combined = torch.cat([word_emb, pos_emb1, pos_emb2], dim=2)  # [B, L, E+2P]
        combined = combined.permute(0, 2, 1)  # [B, C, L]

        # 多尺度卷积
        conv_outs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(combined))  # [B, F, L']
            pooled = conv_out.max(dim=2)[0]  # 全局最大池化
            conv_outs.append(pooled)

        # 特征拼接
        features = torch.cat(conv_outs, dim=1)  # [B, F*K]

        # 实体分类
        entity_logits = self.entity_classifier(features)

        # 关系分类
        relation_logits = self.relation_classifier(features)

        return entity_logits, relation_logits