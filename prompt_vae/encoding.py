import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('../input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models

st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(text):
    return st_model.encode(text)
    
def get_flat_embeddings(text):
    return st_model.encode(text).flatten()
