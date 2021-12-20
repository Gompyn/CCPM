import json
dir_path_1='../CCPM/data/valid.jsonl'
dir_path_2='../CCPM/temp_predict/CCPM_valid.jsonl'

label=[]
pred=[]
for line in open(dir_path_1,'r'):
    item=json.loads(line)
    label.append(item['answer'])

for line in open(dir_path_2,'r'):
    item=json.loads(line)
    pred.append(item['answer'])

score=0
for i in range(len(label)):
    if label[i]==pred[i]:
        score+=1
print(score/len(label))

