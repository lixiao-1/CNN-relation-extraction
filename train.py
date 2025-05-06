import torch
from torch import nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils.evaluate import evaluate
from models.cnn_model import RE_CNN
from utils.data_loader import REDataset
from utils.Data_Conversion import convert_cluener_to_re

# 转换 CLUENER 数据为关系抽取数据
convert_cluener_to_re('data/cluener_public/train.json', 're_train.json')
convert_cluener_to_re('data/cluener_public/dev.json', 're_val.json')

# 初始化组件
# 指定本地模型路径
tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")
train_dataset = REDataset("re_train.json", tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 验证集
val_dataset = REDataset("re_val.json", tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = RE_CNN(
    vocab_size=tokenizer.vocab_size,
    num_classes=len(train_dataset.label2id)
)

# 检查 CUDA 是否可用，并将模型移动到 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练配置
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU 上

# 初始化最佳 F1 分数和最佳模型路径
best_val_f1 = 0
best_model_path = "best_model.pth"

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        # 将所有输入数据移动到 GPU 上
        inputs = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(
            inputs["input_ids"],
            inputs["e1_pos"],
            inputs["e2_pos"]
        )
        loss = criterion(outputs, inputs["label"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader):.4f}")

    # 使用验证集评估模型
    model.eval()
    val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_dataloader, device)
    print(
        f"Epoch {epoch + 1} Validation Evaluation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # 如果当前验证集的 F1 分数高于最佳 F1 分数，则保存当前模型
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at Epoch {epoch + 1} with F1 score: {best_val_f1:.4f}")

# 训练结束后，加载最佳模型
model.load_state_dict(torch.load(best_model_path))
print(f"Loaded best model with F1 score: {best_val_f1:.4f}")