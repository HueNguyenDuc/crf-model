# from optimum.onnxruntime import ORTModelForFeatureExtraction

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd

from BiLSTM_CRF import BiLSTM_CRF

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
print(device)

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
# model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
# model_ort = ORTModelForFeatureExtraction.from_pretrained('BAAI/bge-large-en-v1.5', file_name="onnx/model.onnx")

def prepare_sequence(seq):
    return tokenizer(seq, padding=True, truncation=True, return_tensors='pt').input_ids

sentences = ["this is data for deep purple color"]

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# model_output_ort = model_ort(**encoded_input)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_idx = {
    "O": 0,
    "B": 1,
    "I": 2,
    START_TAG: 3,
    STOP_TAG: 4
}

model = BiLSTM_CRF(tag_to_idx)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

training_data = [(
    "the wall street journal reported today that apple corporation made money",
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia",
    "B I O O O O B".split()
)]


with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0])
    precheck_tags = torch.tensor([tag_to_idx[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence)
        targets = torch.tensor([tag_to_idx[t] for t in tags], dtype=torch.long)

        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0])
    print(model(precheck_sent))