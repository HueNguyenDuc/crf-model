import torch
import torchtext

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
from transformers import AutoModel, AutoTokenizer
from model.model import BiRnnCrf
from product_detection_dataset import TrainingDataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')

mqx_seq = 512

model = BiRnnCrf(tokenizer.vocab_size, 2, 4, 2)
model.cuda(device=device)

training_data = TrainingDataset('data.json', tokenizer=tokenizer, max_seq_len=mqx_seq, device=device)
train_dataloader = DataLoader(training_data, batch_size=30, shuffle=True)

model.load_state_dict(torch.load('colour_saved', weights_only=True))
model.eval()

for batch, data in enumerate(train_dataloader):
    features = data[0]
    masks = data[2]
    vector_token = data[1]
    text = data[3]
    ret = model(features)
    tags = ret[1]
    for i in range(len(tags)):
        print("Text:                    ", text[i])
        print("Predict color position:  ", tags[i])
        print("Real color position:     ", vector_token[i][masks[i]==1].tolist())
        print("\n\n")
    break

a = tokenizer([
    'Apple iPad 7th Gen. 32GB, WiFi, 10.2 in Space Grey Good',
    'Samsung Galaxy Tab S9 11in 128GB WiFi AI Tablet Beige',
    'Amazon Fire HD 10 Lavender 11th Gen 32 GB WiFi 10.1in',
    'vans boys sk8 mid reissue velcro trainers Beige multi'
    ], padding=True, truncation=True, return_tensors='pt')

print(model(a.input_ids.to(device=device)))