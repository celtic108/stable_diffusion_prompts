import pandas as pd
import os


if 'prompt_vae' in os.getcwd():
    base_path = '../data'
else:
    base_path = 'data'

def load_data():
    data = pd.read_parquet(f'{base_path}/prompts/part-00000-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet', columns=['TEXT'])
    return data


train_test_split = 0.95

paths = [f'{base_path}/prompts/train/', f'{base_path}/prompts/test/']
for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)
        data = load_data()
        if 'train' in path:
            data = data.iloc[:int(len(data)*train_test_split)]
            data.to_csv(f'{base_path}/prompts/train/train.csv')
        else:
            data = data.iloc[int(len(data)*train_test_split):]
            data.to_csv(f'{base_path}/prompts/test/test.csv')

def load_train_data():
    return pd.read_csv(f'{base_path}/prompts/train/train.csv')

def load_test_data():
    return pd.read_csv(f'{base_path}/prompts/test/test.csv')
