import torch
from torch import nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils.evaluate import evaluate
from models.cnn_model import RE_CNN
from utils.data_loader import REDataset
import json

# 初始化组件
# 指定本地模型路径
tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")

# 读取 schema 文件获取关系标签信息
label_list = []
try:
    with open("data/duie2.0/schema.json", 'r', encoding='utf-8') as f:
        schema = json.load(f)
        for item in schema:
            label_list.append(item['predicate'])
    label2id = {label: idx for idx, label in enumerate(label_list)}
except FileNotFoundError:
    print("Error: The schema.json file was not found.")
    raise

# 训练集
train_dataset = REDataset("data/duie2.0/train.json", tokenizer, label2id)
if len(train_dataset) == 0:
    print("Error: The training dataset is empty after filtering invalid items. Please check the data.")
else:
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 测试集
    test_dataset = REDataset("data/duie2.0/dev.json", tokenizer, label2id)
    if len(test_dataset) == 0:
        print("Error: The test dataset is empty after filtering invalid items. Please check the data.")
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = RE_CNN(
            vocab_size=tokenizer.vocab_size,
            num_classes=len(label2id)
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
                outputs = model(inputs["input_ids"])
                loss = criterion(outputs, inputs["label"])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader):.4f}")

            # 使用测试集评估模型
            model.eval()
            test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, device)
            print(
                f"Epoch {epoch + 1} Test Evaluation - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

            # 如果当前测试集的 F1 分数高于最佳 F1 分数，则保存当前模型
            if test_f1 > best_val_f1:
                best_val_f1 = test_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at Epoch {epoch + 1} with F1 score: {best_val_f1:.4f}")

        # 训练结束后，加载最佳模型
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model with F1 score: {best_val_f1:.4f}")

        # 使用测试集进行最终评估
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, device)
        print(
            f"Final Test Evaluation - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")