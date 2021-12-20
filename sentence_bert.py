import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
model = SentenceTransformer('paraphrase-mpnet-base-v2') #paraphrase-MiniLM-L6-v2 
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model.max_seq_length = 32

import numpy as np
def cosSim(x,y):
    '''
    余弦相似度
    '''
    x=np.array(x)
    y=np.array(y)
    tmp=np.sum(x*y)
    non=np.linalg.norm(x)*np.linalg.norm(y)
    return np.round(tmp/float(non),20)

dir_path_2='../CCPM/data/valid.jsonl'
dict_list=[]
label=[]
pred=[]
count=0
for line in open(dir_path_2,'r'):
    count+=1
    if count%100==0:
        print(count)
    item=json.loads(line)#存取每一个json文件的内容
    label.append(item['answer'])
    score=[0 for i in range(4)]
    embedding=[]
    for c in item['choices']:
        embedding.append(model.encode(c))
    score[0]=cosSim(embedding[0],embedding[1])+cosSim(embedding[0],embedding[2])+cosSim(embedding[0],embedding[3])
    score[1]=cosSim(embedding[1],embedding[2])+cosSim(embedding[1],embedding[3])+cosSim(embedding[1],embedding[0])
    score[2]=cosSim(embedding[2],embedding[3])+cosSim(embedding[2],embedding[0])+cosSim(embedding[2],embedding[1])
    score[3]=cosSim(embedding[3],embedding[0])+cosSim(embedding[3],embedding[1])+cosSim(embedding[3],embedding[2])
    pred.append(score.index(max(score)))
sum_score=0
for i in range(len(pred)):
    if pred[i]==label[i]:
        sum_score=sum_score+1
print(sum_score)
print(sum_score/len(pred))
    
