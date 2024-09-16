import torch
import torchtext

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
from transformers import AutoModel, AutoTokenizer
from model.model import BiRnnCrf

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
tag_tokenizer = get_tokenizer("basic_english")
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_idx = {
    "O": 0,
}

mqx_seq = 250

model = BiRnnCrf(tokenizer.vocab_size, 3, 4, 2)

sentences = [
    "journal the wall street reported today that apple corporation made money",
    "georgia tech is a university in georgia"
]
tags = [
    "I B I I O O O B I O O",
    "B I O O O O B"
]
tags = [tag_tokenizer(tag) for tag in tags]
def tags_to_vector(tags, max_seq_len=0):
    max_seq_len = max_seq_len if max_seq_len > 0 else len(tags)
    vec = []
    for tag in tags:
        vec.append([tag_to_idx[c.upper()] for c in tag] + [0] * (max_seq_len - len(tag)))
    return vec

tag_vector = tags_to_vector(tags, mqx_seq)
tag_vector = torch.as_tensor(tag_vector)

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

features = encoded_input.input_ids
maskes = encoded_input.attention_mask

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

model.train()
for epoch in range(300):
    model.zero_grad()
    loss = model.loss(features, tag_vector)
    loss.backward()
    optimizer.step()
    print(loss.mean())

model.eval()
print(tag_vector)
print(model(features))