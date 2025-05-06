import json


def convert_cluener_to_re(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_data = []
    for line in lines:
        data = json.loads(line)
        # 假设 data 中有 'entities' 字段，包含实体信息
        entities = data.get('entities', [])
        if len(entities) >= 2:
            # 这里可以根据实际情况判断实体之间的关系
            label = 'has_relation'  # 示例标签
        else:
            label = 'no_relation'
        sample = {
            'text': data['text'],
            'entities': entities,
            'label': label
        }
        new_data.append(sample)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)