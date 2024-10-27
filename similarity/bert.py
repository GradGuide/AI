#!/usr/bin/env python
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def tokenize_sentences(sentences):
    tokens = tokenizer(sentences, return_tensors='pt', padding=True,
                     truncation=True, max_length=512)
    return tokens['input_ids'], tokens['attention_mask']


paragraphs = [
    "The cat is sitting on the table.",
    "The feline is perched on the table.",
]
 
def bert(sentences):
    input_ids, attention_mask = tokenize_sentences(sentences)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    similarity = cosine_similarity([sentence_embeddings[0],sentence_embeddings[1]])
    return similarity
