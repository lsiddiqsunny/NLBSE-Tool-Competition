#!/usr/bin/env python3
# <a href="https://colab.research.google.com/github/lsiddiqsunny/NLBSE-Tool-Competition/blob/master/Text_Classification_With_BERT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

# %%
import os
import numpy as np
import pandas as pd
import gensim
from tqdm.auto import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup

print("Included Library")

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
else:
    exit(0)
# %%
# persistent file storage
# https://colab.research.google.com/notebooks/io.ipynb
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# if not os.path.isfile("github-labels-top3-803k-train.csv"):
#   !curl "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-train.tar.gz" | tar -xz

df = pd.read_csv('~/Public/NLBSE-Tool-Competition/github-labels-top3-803k-train.csv')

# %%
print(df.head())

# %%
# if not os.path.isfile("github-labels-top3-803k-test.csv"):
#   !curl "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-test.tar.gz" | tar -xz

testdf = pd.read_csv("~/Public/NLBSE-Tool-Competition/github-labels-top3-803k-test.csv")
testdf.groupby("issue_label").size()

# %%
print(df['issue_label'].value_counts())

# %%
possible_labels = df.issue_label.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
print(label_dict)

# %%
df['label'] = df.issue_label.replace(label_dict)
testdf['label'] = testdf.issue_label.replace(label_dict)

# %%
# preprocessing can be customized by participants
def preprocess(row):
  # concatenate title and body, then remove whitespaces
  doc = ""
  doc += str(row.issue_title)
  doc += " "
  doc += str(row.issue_body)
  # https://radimrehurek.com/gensim/parsing/preprocessing.html
  doc = gensim.parsing.preprocessing.strip_multiple_whitespaces(doc)
  return doc

# %%
df['issue_data'] = df.apply(preprocess, axis=1)

newDF = df[['issue_label','issue_data','label']]
print(newDF.head())

# %%
df = newDF.copy()
print(df.head())

# %%

testdf['issue_data'] = testdf.apply(preprocess, axis=1)

newTestDF = testdf[['issue_label','issue_data','label']]
print(newTestDF.head())

# %%
testdf = newTestDF.copy()
print(testdf.head())

# %%
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=df.label.values)

# %%
df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

# %%
df.groupby(['issue_label', 'label', 'data_type']).count()

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

# %%
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].issue_data.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='longest',
    truncation=True, 
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

print("Done train encoding")
# %%
encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].issue_data.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='longest',
    truncation=True,
    return_tensors='pt'
)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

print("Done val encoding")
# %%
encoded_data_test = tokenizer.batch_encode_plus(
    testdf.issue_data.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='longest',
    truncation=True,
    return_tensors='pt'
)

input_ids_test= encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test= torch.tensor(testdf.label.values)

print("Done test encoding")
# %%
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

# %%
print(len(dataset_train), len(dataset_val), len(dataset_test))

# %%
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)


dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=batch_size)

# %%

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)

# %%
epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

print("Startig training")
# %%
from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

# %%
import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# %%


model.to(device)
# %%
def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

# %%
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

# %%
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

# %%
model.load_state_dict(torch.load('finetuned_BERT_epoch_10.model', map_location=device))

# %%
_, predictions, true_vals = evaluate(dataloader_test)

# %%
accuracy_per_class(predictions, true_vals)

# %%
f1_score = f1_score_func(predictions, true_vals)
print(f1_score)


