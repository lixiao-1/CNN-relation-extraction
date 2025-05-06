import json

def convert_cluener_to_re(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_data = []
    for line in lines:
        data = json.loads(line)
        text = data['text']
        label_info = data.get('label', {})
        entities = []
        for entity_type, entity_dict in label_info.items():
            for entity_name, positions in entity_dict.items():
                for start, end in positions:
                    entities.append({
                        'name': entity_name,
                        'type': entity_type,
                        'start': start,
                        'end': end
                    })
        # 假设每个样本只有一个主要的标签类型，这里简单取第一个实体的类型作为标签
        label = entities[0]['type'] if entities else 'no_relation'
        sample = {
            'text': text,
            'entities': entities,
            'label': label
        }
        new_data.append(sample)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)