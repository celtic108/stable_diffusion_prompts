from loader import load_data
from encoding import get_embeddings
import numpy as np
import pickle
from tqdm import tqdm
import os

prompt_path = '../data/prompts/embedded/'
if not os.path.exists(prompt_path):
    os.mkdir(prompt_path)
        
file_name = 'embedded_prompts_{}.pkl'

tqdm.pandas()

data = load_data()
print(data.shape)

cache_for_gpu = []
cache_for_file_out = []
file_counter = 0
for path in os.listdir(prompt_path):
    if os.path.isfile(os.path.join(prompt_path, path)):
        file_counter += 1


for i, line in tqdm(data.iterrows(), total=data.shape[0]):
    cache_for_gpu.append('' if line['TEXT'] is None else line['TEXT'])
    
    if len(cache_for_gpu) == 700:
        cache_for_file_out.append(get_embeddings(cache_for_gpu))
        cache_for_gpu = []
    
    if len(cache_for_file_out) > 40:
        cache_for_file_out = np.concatenate(cache_for_file_out, axis=0)
        with open(os.path.join(prompt_path, file_name.format(file_counter)), 'wb') as f:
            pickle.dump(cache_for_file_out, f)
        file_counter += 1
        cache_for_file_out = []
        
if len(cache_for_gpu) > 0:
    cache_for_file_out.append(get_embeddings(cache_for_gpu))
    cache_for_gpu = []
    
if len(cache_for_file_out) > 0:
    cache_for_file_out = np.concatenate(cache_for_file_out, axis=0)
    with open(os.path.join(prompt_path, file_name.format(file_counter)), 'wb') as f:
        pickle.dump(cache_for_file_out, f)
    file_counter += 1
    cache_for_file_out = []
# print(data.head())

# matrix = data['embeddings'].to_list()
# matrix = np.array(matrix)
# print(matrix.shape)

# with open(prompt_path, 'wb') as f:
#     pickle.dump(matrix, f)