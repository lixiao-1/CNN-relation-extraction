import json

def fix_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 尝试解析 JSON
        data = json.loads(content)
        print("JSON 文件格式正确。")
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        # 尝试添加方括号来修正格式
        try:
            content = '[' + content.replace('}\n{', '},{') + ']'
            data = json.loads(content)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print("JSON 文件格式已修正。")
        except json.JSONDecodeError as e:
            print(f"无法修正 JSON 文件格式: {e}")

# 使用示例
fix_json_file("data/duie2.0/test2.json")