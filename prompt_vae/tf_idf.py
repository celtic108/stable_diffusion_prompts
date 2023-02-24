import json
from loader import load_data
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np


path = 'idf_values.json'
try:
    with open(path, 'r') as f:
        idf = json.load(f)
    max_val = -float('inf')
    min_val = float('inf')
    for i, key in enumerate(idf):
        max_val = max([max_val, idf[key]])
        min_val = min([min_val, idf[key]])
    print(min_val, max_val)
except:
    print('Could not locate file')
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    text = load_data()

    idf = {}

    tokenizer_queue = []
    for i, row in tqdm(text.iterrows(), total = text.shape[0]):
        # if len(tokenizer_queue) < 10:
        #     tokenizer_queue.append(row['TEXT'])
        # else:
        #     for t in tokenizer_queue:
        #         print(type(t))
        #     print(type(tokenizer_queue[0]))
        t = row['TEXT'] if row['TEXT'] is not None else ''
        tokens = tokenizer.encode_plus(t, padding=True, truncation=True, return_tensors='pt')['input_ids'].numpy()[0]
        for token in list(set(list(tokens))):
            idf[int(token)] = idf.get(int(token), 0) + 1
        

    for token in idf:
        idf[token] =  float(np.log((1 + text.shape[0]) / (1 + idf[token])))
    
    with open(path, 'w') as f:
        json.dump(idf, f)