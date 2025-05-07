import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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

    # 计算 ROC 曲线和 AUC
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_classes = all_probs.shape[1]
    for i in range(num_classes):
        if np.sum(all_labels == i) > 0:  # 检查该类别是否有正样本
            fpr[i], tpr[i], _ = roc_curve(all_labels, all_probs[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

    return accuracy, precision, recall, f1, fpr, tpr, roc_auc