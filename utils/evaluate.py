import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 新增：用于记录预测概率

    with torch.no_grad():
        for batch in dataloader:
            # 将输入数据移动到 GPU 上
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(inputs["input_ids"], inputs["e1_pos"], inputs["e2_pos"])
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)  # 计算预测概率

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(inputs["label"].cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算 PR 曲线所需的指标
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    precision_dict = dict()
    recall_dict = dict()
    pr_auc_dict = dict()
    num_classes = all_probs.shape[1]
    for i in range(num_classes):
        if np.sum(all_labels == i) > 0:  # 检查该类别是否有正样本
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(all_labels, all_probs[:, i], pos_label=i)
            pr_auc_dict[i] = auc(recall_dict[i], precision_dict[i])

    return accuracy, precision, recall, f1, precision_dict, recall_dict, pr_auc_dict