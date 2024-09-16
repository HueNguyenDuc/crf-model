import torch
import json

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

tag_to_idx = {
    "O": 0,
    "COLOUR": 1,
}

def extend_data(data, max_seq_len, device):
  data_len = len(data)
  if max_seq_len > data_len:
    final_data = torch.cat([data, torch.full((max_seq_len - data_len, ), 0, dtype=torch.long, device=device)])
  elif max_seq_len < data_len:
    final_data = data[:max_seq_len]
  else:
    final_data = data
  return torch.as_tensor(final_data)
    

def tags_to_vector(tag, max_seq_len):
    tag_len = len(tag)
    if max_seq_len > tag_len:
      tag_token = [tag_to_idx[c.upper()] for c in tag] + [0] * (max_seq_len - tag_len)
    elif max_seq_len < tag_len:
      tag_token = [tag_to_idx[c.upper()] for c in tag[:max_seq_len]]
    else:
      tag_token = [tag_to_idx[c.upper()] for c in tag]
    return torch.as_tensor(tag_token)

class TrainingDataset(torch.utils.data.Dataset):
  def __init__(self, file_name, max_seq_len, tokenizer, device=None):
    self.device = device if device else torch.device("cpu:0")
    self.max_seq_len = max_seq_len
    self.tokenizer = tokenizer
    f = open(file_name, encoding="utf8")
    self.data = json.load(f)
    f.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    
    encoded_input = self.tokenizer([item['text']], padding=True, truncation=True, return_tensors='pt')
    text_len = len(encoded_input.input_ids[0])

    encoded_token = encoded_input.input_ids[0].to(device=self.device)
    if self.max_seq_len > text_len:
      extend_token = torch.full((self.max_seq_len - text_len,), 0, dtype=torch.long, device=self.device)
      encoded_token = torch.cat([encoded_token, extend_token])
    elif self.max_seq_len < text_len:
      encoded_token = encoded_token[:self.max_seq_len]

    tag = tags_to_vector(item['tag'].split(), max_seq_len=self.max_seq_len)
    mask = extend_data(encoded_input['attention_mask'][0].to(device=self.device), self.max_seq_len, self.device)

    return (
      encoded_token, 
      tag.to(device=self.device), 
      mask,
      item['text']
    )
  
# tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
# training_data = TrainingDataset('data.json', tokenizer=tokenizer, max_seq_len=512)
# train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True)

# print(next(iter(train_dataloader)))