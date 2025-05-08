import tkinter as tk
from tkinter import messagebox
import torch
from transformers import BertTokenizer
from models.cnn_model import RE_CNN
import json

# 加载模型和标签映射
tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")
label_list = []
try:
    with open("data/duie2.0/schema.json", 'r', encoding='utf-8') as f:
        schema = json.load(f)
        for item in schema:
            label_list.append(item['predicate'])
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
except FileNotFoundError:
    messagebox.showerror("错误", "未找到 schema.json 文件！")
    raise

# 加载模型
model = RE_CNN(
    vocab_size=tokenizer.vocab_size,
    num_classes=len(label2id)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
except FileNotFoundError:
    messagebox.showerror("错误", "未找到最佳模型文件！")
    raise


def predict_text():
    text = entry.get()
    if not text:
        messagebox.showwarning("警告", "请输入待识别的文本！")
        return
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        preds = torch.argmax(outputs, dim=1)
        predicted_label = id2label[preds.item()]
    result_label.config(text=f"预测结果: {predicted_label}")


# 创建主窗口
root = tk.Tk()
root.title("关系抽取识别")

# 创建输入框
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# 创建识别按钮
button = tk.Button(root, text="识别", command=predict_text)
button.pack(pady=5)

# 创建结果显示标签
result_label = tk.Label(root, text="预测结果: ")
result_label.pack(pady=10)

# 运行主循环
root.mainloop()