import numpy as np 
import pandas as pd 
import os
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torchvision
from prompt_vae.match_embeddings import find_embedding


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EmbeddingLayers(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLayers, self).__init__()
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, 384)
        self.fc3 = torch.nn.Linear(384, 384)
        
    def forward(self, x):
        x = x[:,0]
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
feature_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384').to(device)

embedding_layers = EmbeddingLayers().to(device)
embedding_layers.load_state_dict(torch.load('embedding_model_20230223_203227_11'))
embedding_layers.eval()

header = 'imgId_eId,val\n'
image_ids = []
values = []
for dirname, _, filenames in os.walk('data/images'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        if 'generated' not in file_path:
            image = torchvision.io.read_image(file_path)
            image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
            print(image.shape)
            with torch.no_grad():
                outputs = feature_model(pixel_values = image, output_hidden_states = True, return_dict = True)
            state = outputs.hidden_states[-1]
            final_output = embedding_layers(state)#.item()#.numpy()
            # ground_truth = find_embedding(int(filename.split('.png')[0].split('_')[-1]))
            for i, v in enumerate(final_output.cpu().detach().numpy().flatten()):
                image_ids.append(f"{filename.split('.')[0]}_{i}")
                values.append(v)

with open('submission.csv', 'w') as f:
    f.write(header)
    for id, v in zip(image_ids, values):
        f.write(','.join([id, str(v) + '\n']))

answers = {}    
with open('submission.csv', 'r') as f:
    for i, line in enumerate(f):
        if i > 0:
            img_id, val = line.strip('\n').split(',')
            val = float(val)
            index = int(img_id.split('.png')[0].split('_')[-1])
            img_name = img_id.split('.png')[0].split('_')[0]
            assert len(answers.get(img_name, [])) == index
            answers[img_name] = answers.get(img_name, []) + [val]

targets = {}
with open('data/sample_submission.csv', 'r') as f:
    for i, line in enumerate(f):
        if i > 0:
            img_id, val = line.strip('\n').split(',')
            val = float(val)
            index = int(img_id.split('.png')[0].split('_')[-1])
            img_name = img_id.split('.png')[0].split('_')[0]
            assert len(targets.get(img_name, [])) == index
            targets[img_name] = targets.get(img_name, []) + [val]
        
for key in answers:
    print(key, answers[key][:10])

for key in targets:
    print(key, targets[key][:10]) 
        
for key in answers:
    a = np.array(answers[key])
    b = np.array(targets[key])
    print(f"COSINE SIM: {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))}")     
