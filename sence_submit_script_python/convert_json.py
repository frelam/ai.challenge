import numpy as np
import json
from pprint import pprint
files_name=np.load('/home/frelam/PycharmProjects/sence_submit_script_python/files_name.npy')
top3=np.load('/home/frelam/PycharmProjects/sence_submit_script_python/top3.npy')

aaa = []

for i in range(len(files_name)):
    temp = {"image_id": files_name[i], "label_id": top3[i].tolist()}
    aaa.append(temp)
print i
print len(aaa)
with open('submit.json', 'w') as f:
    json.dump(aaa, f, ensure_ascii=False)

with open('/home/frelam/PycharmProjects/sence_submit_script_python/submit.json','r') as f:
    data = json.load(f)

new = json.dumps(data,ensure_ascii=False)
pprint(new)