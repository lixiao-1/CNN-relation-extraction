import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from models.cnn_model import RE_CNN
from utils.data_loader import REDataset
from utils.Data_Conversion import convert_cluener_to_re
import json

# 初始化组件
# 指定本地模型路径
tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")
model = RE_CNN(
    vocab_size=tokenizer.vocab_size,
    num_classes=len(REDataset("re_test.json", tokenizer).label2id)
)

# 检查 CUDA 是否可用，并将模型移动到 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载最佳模型
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()


def predict_from_file(file_path):
    # 转换待预测数据
    convert_cluener_to_re(file_path, 're_test.json')

    predict_dataset = REDataset("re_test.json", tokenizer)
    predict_dataloader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

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

    result_text = ""
    for i, data in enumerate(predict_data):
        result_text += f"Text: {data['text']}\n"
        result_text += f"Predicted Label: {predict_labels[i]}\n"
        result_text += "-" * 50 + "\n"
    return result_text


def predict_from_text():
    text = input_text.get("1.0", tk.END).strip()
    if text:
        # 构造符合 REDataset 要求的数据格式（必须包含 entities 字段）
        re_data = [{
            "text": text,
            "label": "no_relation",
            "entities": [
                {"text": "", "start": -1, "end": -1},  # head 实体占位
                {"text": "", "start": -1, "end": -1}  # tail 实体占位
            ]
        }]

        # 保存为 re_test.json
        with open('re_test.json', 'w', encoding='utf-8') as f:
            json.dump(re_data, f, ensure_ascii=False, indent=4)

        # 加载数据并预测
        predict_dataset = REDataset("re_test.json", tokenizer)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch in predict_dataloader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(inputs["input_ids"], inputs["e1_pos"], inputs["e2_pos"])
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())

        id2label = {v: k for k, v in predict_dataset.label2id.items()}
        predict_labels = [id2label[pred] for pred in all_preds]

        # 显示结果
        result_text = f"Text: {text}\nPredicted Label: {predict_labels[0]}\n" + "-" * 50 + "\n"
        result_text_area.delete(1.0, tk.END)
        result_text_area.insert(tk.END, result_text)
    else:
        messagebox.showwarning("Warning", "请输入文本！")


def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            result = predict_from_file(file_path)
            result_text_area.delete(1.0, tk.END)
            result_text_area.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# 创建 GUI 窗口
root = tk.Tk()
root.title("Relation Extraction Prediction")

# 创建输入框用于输入文本
input_text = tk.Text(root, height=5, width=80)
input_text.pack(pady=10)

# 创建确定按钮用于触发文本预测
predict_text_button = tk.Button(root, text="Predict Text", command=predict_from_text)
predict_text_button.pack(pady=5)

# 创建按钮选择文件
select_file_button = tk.Button(root, text="Select File", command=select_file)
select_file_button.pack(pady=10)

# 创建文本框显示结果
result_text_area = tk.Text(root, height=20, width=80)
result_text_area.pack(pady=10)

# 运行 GUI 主循环
root.mainloop()
