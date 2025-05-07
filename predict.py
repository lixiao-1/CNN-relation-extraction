import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from models.cnn_model import RE_CNN
from utils.data_loader import REDataset
from utils.Data_Conversion import convert_cluener_to_re
import json

# 转换待预测数据
convert_cluener_to_re('data/cluener_public/test.json', 're_test.json')

# 初始化组件
# 指定本地模型路径
tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")
predict_dataset = REDataset("re_test.json", tokenizer)
predict_dataloader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

model = RE_CNN(
    vocab_size=tokenizer.vocab_size,
    num_classes=len(predict_dataset.label2id)
)

# 检查 CUDA 是否可用，并将模型移动到 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载最佳模型
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 预测
all_preds = []
with torch.no_grad():
    for batch in predict_dataloader:
        # 将输入数据移动到 GPU 上
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(inputs["input_ids"], inputs["e1_pos"], inputs["e2_pos"])
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())

# 将预测结果转换为标签
id2label = {v: k for k, v in predict_dataset.label2id.items()}
predict_labels = [id2label[pred] for pred in all_preds]

# 输出预测结果
with open('re_test.json', 'r', encoding='utf-8') as f:
    predict_data = json.load(f)

for i, data in enumerate(predict_data):
    print(f"Text: {data['text']}")
    print(f"Predicted Label: {predict_labels[i]}")
    print("-" * 50)