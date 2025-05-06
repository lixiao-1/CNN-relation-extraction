import json

def convert_cluener_to_re(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_data = []
    for line in lines:
        data = json.loads(line)
        # 假设原数据中有 'label' 字段，直接使用该字段作为标签
        label = data.get('label', 'no_relation')  # 如果原数据中没有 'label' 字段，默认使用 'no_relation'
        sample = {
            'text': data['text'],
            'entities': data.get('entities', []),
            'label': label
        }
        new_data.append(sample)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)