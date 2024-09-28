import json

with open('./data/c/log_data.json', 'r') as f:
    log_data = json.load(f)

max_len = 0
for log in log_data:
    if log['log_num'] > max_len:
        max_len = log['log_num']
print(max_len)
