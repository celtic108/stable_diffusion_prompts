from translate_encoding import TranslationLayers
import loader
import torch
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

translation_layers = TranslationLayers()
translation_layers.load_state_dict(torch.load('model_20230221_051654_0'))
translation_layers.eval()
translation_layers = translation_layers.to(device)

data = loader.load_data()

text = data.iloc[0]['TEXT']
# text = ['Blue Beach Umbrellas, Point Of Rocks, Crescent Beach, Siesta Key - Spiral Notebook', 'It was the best of times, it was the worst of times']

for i, row in data.iterrows():
    text = row['TEXT']
    print(text)

    encoded_input = tokenizer((text), padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    final_output = translation_layers(model_output['pooler_output'])# * encoded_input['attention_mask'][:, :, None]

    # preds = torch.nn.functional.softmax(final_output, dim=-1).argmax(dim=-1)
    preds = torch.where(final_output > 4, 1, 0).nonzero().cpu().detach().numpy()
    pred_list = []
    for p in preds:
        if len(pred_list) == p[0]:
            pred_list.append([])
        pred_list[p[0]].append(p[1])
    print(tokenizer.batch_decode(sequences = pred_list, skip_special_tokens=True))
    print("*"*30)
    if i > 10:
        break