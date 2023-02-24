import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from loader import load_data
from match_embeddings import find_embedding


tokenizer = AutoTokenizer.from_pretrained("gpt2")

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        return find_embedding(index)

    def __len__(self):
        return len(self.dataframe)

data = CustomDataset(dataframe= load_data())
dataloader = DataLoader(data)

text_data = load_data()

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(128, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 128)
#         self.fc4 = nn.Linear(128, 128)
#         self.fc5 = nn.Linear(128, 768 * 20)

#     def forward(self, x):
#         x = torch.nn.functional.relu(self.fc1(x))
#         x = torch.nn.functional.relu(self.fc2(x))
#         x = torch.nn.functional.relu(self.fc3(x))
#         x = torch.nn.functional.relu(self.fc4(x))
#         x = self.fc5(x)
#         x = torch.reshape(x, (-1, 20, 768))

# generator = Generator()

# loss_fn = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam()

decoder_model = AutoModelForCausalLM.from_pretrained('gpt2')

# def train_one_epoch(epoch_index, tb_writer)
#     running_loss = 0.
#     last_loss = 0.

#     for i, true_embedding in enumerate(data_loader):
#         optimizer.zero_grad()

#         outputs = model(inputs)

#         # Compute the loss and its gradients
#         loss = loss_fn(outputs, labels)
#         loss.backward()

#         # Adjust learning weights
#         optimizer.step()

#         # Gather data and report
#         running_loss += loss.item()
#         if i % 1000 == 999:
#             last_loss = running_loss / 1000 # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             tb_x = epoch_index * len(training_loader) + i + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.

#     return last_loss

# random_input = np.random.rand(1,20,768)*20

for i, random_input in enumerate(dataloader):
    print(i)
    print(random_input[0,:10])
    print(text_data.iloc[i]['TEXT'])
    print(find_embedding(i)[:10])
    # inputs = tokenizer("It was the best of times, it was the worst of times", return_tensors="pt")
    # print(inputs)

    # outputs = model(**inputs)#, labels=inputs["input_ids"])
    outputs = decoder_model(inputs_embeds=random_input.bfloat16())

    # loss = outputs.loss
    logits = outputs.logits#.detach().numpy()

    # print(type(outputs))
    # print(outputs.keys())
    # print(loss)
    # print(logits)
    # print(logits.shape)

    preds = torch.nn.functional.softmax(logits, dim=-1).argmax(dim=-1)

    print(preds)

    print(tokenizer.batch_decode(sequences = preds, skip_special_tokens = True))
    if i == 10:
        break