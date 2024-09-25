import json

with open('log_data_all.json', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    print(item)
    item['user_id'] += 1
    for log in item['logs']:
        log['exer_id'] += 1
        log['knowledge_code'] = [i+1 for i in log['knowledge_code']]
    print(item)

with open('log_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

