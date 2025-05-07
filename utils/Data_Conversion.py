import json

def convert_duie_to_re(input_file, output_file):
    new_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    text = sample['text']
                    spo_list = sample['spo_list']
                    entities = []
                    relations = []
                    for spo in spo_list:
                        subject = spo['subject']
                        subject_type = spo['subject_type']
                        object_ = spo['object']['@value']
                        object_type = spo['object_type']['@value']
                        predicate = spo['predicate']

                        subject_start = text.find(subject)
                        object_start = text.find(object_)

                        if subject_start != -1 and object_start != -1:
                            entities.append({
                                'name': subject,
                                'type': subject_type,
                                'start': subject_start,
                                'end': subject_start + len(subject)
                            })
                            entities.append({
                                'name': object_,
                                'type': object_type,
                                'start': object_start,
                                'end': object_start + len(object_)
                            })
                            relations.append({
                                'subject': subject,
                                'object': object_,
                                'predicate': predicate
                            })

                    # 假设每个样本只有一个主要的标签类型，这里简单取第一个实体的类型作为标签
                    label = entities[0]['type'] if entities else 'no_relation'
                    new_sample = {
                        'text': text,
                        'entities': entities,
                        'relations': relations,
                        'label': label
                    }
                    new_data.append(new_sample)
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}. Error: {e}")
    except FileNotFoundError:
        print(f"File {input_file} not found.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)