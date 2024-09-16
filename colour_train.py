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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

training_data = TrainingDataset('data.json', tokenizer=tokenizer, max_seq_len=mqx_seq, device=device)
train_dataloader = DataLoader(training_data, batch_size=30, shuffle=True)

model.train()
for epoch in range(500):
    batch_loss = []
    for batch, data in enumerate(train_dataloader):
        model.zero_grad()
        features = data[0]
        tag_vector = data[1]
        loss = model.loss(features, tag_vector)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.mean())
    batch_loss = torch.stack(batch_loss)
    print(epoch, " ----- ", batch_loss.mean())

model.eval()

torch.save(model.state_dict(), 'colour_saved')

for batch, data in enumerate(train_dataloader):
    features = data[0]
    vector_token = data[1]
    ret = model(features)
    print(vector_token)
    print(ret[1])
    break

