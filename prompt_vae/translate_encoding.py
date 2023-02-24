import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.utils.data import Dataset, DataLoader
import loader
from match_embeddings import find_embedding
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time


num_classes = 30522

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.text_data = dataframe

    def __getitem__(self, index):
        text = self.text_data.iloc[index]['TEXT']
        if isinstance(text, str):
            return text
        else:
            return ''

    def __len__(self):
        return len(self.text_data)

class TranslationLayers(torch.nn.Module):
    def __init__(self):
        super(TranslationLayers, self).__init__()
        self.fc1 = torch.nn.Linear(384, 384)
        self.fc2 = torch.nn.Linear(384, 384)
        self.fc3 = torch.nn.Linear(384, 384)
        self.fc4 = torch.nn.Linear(384, 30522)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        return x

class ToOneHot(torch.nn.Module):
    def __init__(self):
        super(ToOneHot, self).__init__()
    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=num_classes)

class TargetEncoder(torch.nn.Module):
    def __init__(self):
        super(TargetEncoder, self).__init__()
        self.oh = ToOneHot()
                
    def forward(self, x, weights):
        x = self.oh(x['input_ids'])
        x = x * weights
        x = torch.max(x, dim=1)[0]
        x = x.bfloat16()
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    import tf_idf

    weights = [[[]]]
    for i in range(num_classes):
        weights[0][0].append(tf_idf.idf.get(str(i), 0) / tf_idf.max_val)
    weights = torch.FloatTensor(weights).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', model_max_length = 256)
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

    train_data = CustomDataset(dataframe = loader.load_train_data())
    train_dataloader = DataLoader(train_data, batch_size=60, shuffle=True)
    print(f"TRAIN LEN: {len(train_dataloader)}")

    test_data = CustomDataset(dataframe = loader.load_test_data())
    test_dataloader = DataLoader(test_data, batch_size=60)
    print(f"TEST LEN: {len(test_dataloader)}")

    translation_layers = TranslationLayers().to(device)

    target_encoder = TargetEncoder().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(translation_layers.parameters(), lr = 0.0002)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            try:
                encoded_input = tokenizer((data), padding=True, truncation=True, return_tensors='pt')
            except TypeError:
                print(data)
                assert False
            encoded_input = encoded_input.to(device)
            target = target_encoder(encoded_input, weights)
            with torch.no_grad():
                model_output = model(**encoded_input)
            final_output = translation_layers(model_output['pooler_output'])# * encoded_input['attention_mask'][:, :, None]
            final_output = final_output.bfloat16()
            loss = loss_fn(final_output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print(f'  batch {i + 1} loss: {last_loss}')
                tb_x = epoch_index * len(train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 1

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        start_time = time.time()
        avg_loss = train_one_epoch(epoch_number, writer)
        print(f"Trained in {time.time()-start_time:0.4f} seconds")

        model.train(False)

        start_time = time.time()
        running_vloss = 0.0
        for i, vdata in enumerate(test_dataloader):
            encoded_input = tokenizer((vdata), padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(device)
            target = target_encoder(encoded_input, weights)
            with torch.no_grad():
                model_output = model(**encoded_input)
            final_output = translation_layers(model_output['pooler_output'])# * encoded_input['attention_mask'][:, :, None]
            final_output = final_output.bfloat16()
            vloss = loss_fn(final_output, target).item()
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
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(translation_layers.state_dict(), model_path)

        epoch_number += 1

if __name__ == '__main__':
    main()