import json
with open('test.json', 'r', encoding='utf-8') as f:
    y = json.load(f)
a = list()
for i in y:
    a.append([i, y[i]])
print(a)