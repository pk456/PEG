result = []
with open('knowledgeGraph.txt', 'r') as f:
    for i in f.readlines():
        i = i.replace('\n', '').split('\t')
        # 给i每个元素+1
        i = [int(j) + 1 for j in i]
        i = str(i[0]) + '\t' + str(i[1]) + '\n'
        result.append(i)
print(len(result))
with open('knowledgeGraph.txt', 'w') as f:
    for r in result:
        f.write(r)