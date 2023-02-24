from prompt_vae import loader, match_embeddings
import torchvision
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

path = 'data/images/generated_images/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
number_of_files = len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))])
train_size = int(0.8 * number_of_files)

class TrainDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __getitem__(self, index):
        embedding = match_embeddings.find_embedding(index)
        image = torchvision.io.read_image(f'{self.path}/{index}.png')
        return image, embedding
        
    def __len__(self):
        return train_size

class TestDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __getitem__(self, index):
        embedding = match_embeddings.find_embedding(index + train_size)
        image = torchvision.io.read_image(f'{self.path}{index + train_size}.png')
        return image, embedding
        
    def __len__(self):
        return number_of_files - train_size

class EmbeddingLayers(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLayers, self).__init__()
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, 512)
        self.fc3 = torch.nn.Linear(512, 384)
        self.fc4 = torch.nn.Linear(384, 384)
        self.fc5 = torch.nn.Linear(384, 384)

        self.norm1 = torch.nn.BatchNorm1d(768)
        self.norm2 = torch.nn.BatchNorm1d(512)
        self.norm3 = torch.nn.BatchNorm1d(384)
        self.norm4 = torch.nn.BatchNorm1d(384)
        
    def forward(self, x):
        x = x[:,0]
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.norm1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.norm2(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.norm3(x)
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.norm4(x)
        x = torch.tanh(self.fc5(x))
        return x

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.zeros_(m.bias.data)


def main():
    train_data = TrainDataset(path)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print(f"TRAIN LEN: {len(train_dataloader)}")

    test_data = TestDataset(path)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print(f"TEST LEN: {len(test_dataloader)}")

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    feature_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384').to(device)

    embedding_layers = EmbeddingLayers().to(device)
    embedding_layers.apply(weights_init)

    loss_fn = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(embedding_layers.parameters(), lr = 0.0001)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, (image_batch, encoding_batch) in enumerate(train_dataloader):
            images = []
            optimizer.zero_grad()
            encoding_batch = encoding_batch.to(device)
            for image in image_batch:
                images.append(feature_extractor(images=image, return_tensors="pt").pixel_values)
            images = torch.concat(images, dim=0).to(device)
            with torch.no_grad():
                outputs = feature_model(pixel_values = images, output_hidden_states = True, return_dict = True)
            state = outputs.hidden_states[-1]
            final_output = embedding_layers(state)
            loss = loss_fn(final_output, encoding_batch)
            print(loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i == len(train_dataloader) - 1:
                last_loss = running_loss / (i + 1)
                print(f'  batch {i + 1} loss: {last_loss}')
                tb_x = epoch_index * len(train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('embedding_layers_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        embedding_layers.train(True)
        start_time = time.time()
        avg_loss = train_one_epoch(epoch_number, writer)
        print(f"Trained in {time.time()-start_time:0.4f} seconds")

        embedding_layers.train(False)

        start_time = time.time()
        running_vloss = 0.0
        for i, (image_batch, encoding_batch) in enumerate(test_dataloader):
            encoding_batch = encoding_batch.to(device)
            images = []
            for image in image_batch:
                images.append(feature_extractor(images=image, return_tensors="pt").pixel_values)
            images = torch.concat(images, dim=0).to(device)
            with torch.no_grad():
                outputs = feature_model(pixel_values = images, output_hidden_states = True, return_dict = True)
            state = outputs.hidden_states[-1]
            final_output = embedding_layers(state)
            vloss = loss_fn(final_output, encoding_batch)
            vloss = vloss.item()
            running_vloss += vloss
        print(f"Validated in {time.time()-start_time:0.4f} seconds")

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'embedding_model_{}_{}'.format(timestamp, epoch_number)
            torch.save(embedding_layers.state_dict(), model_path)

        epoch_number += 1


    header = 'imgId_eId,val\n'
    image_ids = []
    values = []
    for dirname, _, filenames in os.walk('data/images'):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            if 'generated' not in file_path:
                image = torchvision.io.read_image(file_path)
                image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    outputs = feature_model(pixel_values = image, output_hidden_states = True, return_dict = True)
                state = outputs.hidden_states[-1]
                final_output = embedding_layers(state)
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
        print(np.mean(targets[key]))
        print(np.std(targets[key]))
        print(np.max(targets[key]))
        print(np.min(targets[key]))
        
            

if __name__ == '__main__':
    main()
