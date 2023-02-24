from random import randint
import pickle
import numpy as np
import os

if 'prompt_vae' in os.getcwd():
    base_path = '../data'
    from encoding import get_embeddings
    from loader import load_data
else:
    base_path = 'data'
    from prompt_vae.encoding import get_embeddings
    from prompt_vae.loader import load_data


def find_embedding(idx):
    file_number = idx // 28700
    line_number = idx % 28700
    with open(f'{base_path}/prompts/embedded/embedded_prompts_{file_number}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data[line_number]    

if __name__ == '__main__':
    text_data = load_data()

    for i in range(695, 705):
        r = randint(0, 1000000)
        print(r)
        found_embedding = find_embedding(r)
        print(text_data.iloc[r])
        embedding = get_embeddings(text_data['TEXT'].iloc[r])
        # print(found_embedding.shape)
        # print(embedding.shape)
        print(found_embedding[:10])
        print(embedding[:10])
        assert np.allclose(found_embedding, embedding, atol=1e-6)


