import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('../input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models

comp_path = Path('data/')

prompts = pd.read_csv(comp_path / 'prompts.csv', index_col='imgId')
print(prompts.head(7))

print('*'*30)
sample_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='imgId_eId')
print(sample_submission.head())

st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

prompt_embeddings = st_model.encode(prompts['prompt']).flatten()

assert np.all(np.isclose(sample_submission['val'].values, prompt_embeddings, atol=1e-07))

print("YES, EVERYTHING CHECKS OUT")