from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

sentences = ['This is an example sentence', 'Each sentence is converted']

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('nreimers/MiniLM-L6-H384-uncased')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print(encoded_input['input_ids'])

with torch.no_grad():
    model_output = model(**encoded_input)

print(model_output.keys())
print(model_output['pooler_output'].shape)

preds = torch.nn.functional.softmax(model_output['last_hidden_state'], dim=-1).argmax(dim=-1)

print(preds)

print(tokenizer.batch_decode(sequences = preds, skip_special_tokens = True))

print(tokenizer.vocab_files_names)