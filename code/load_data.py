import pickle as pickle
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def dict_label_to_num(x):
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  return dict_label_to_num[x]

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

# Typerd entity marker(punct) to Query and Sentence
def preprocessing_dataset_with_punc_sentence12(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  sentence = []
  for i,j,k in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    S_WORD = i[1:-1].split(", '")[0].split(': ')[1][1:-1]
    S_TYPE = i[1:-1].split(", '")[-1].split(': ')[1][1:-1]    
    S_TEMP = ''.join(['@', '+', S_TYPE, '+', S_WORD, '@'])
    subject_entity.append(S_TEMP)
    
    O_WORD = j[1:-1].split(", '")[0].split(': ')[1][1:-1]
    O_TYPE = j[1:-1].split(", '")[-1].split(': ')[1][1:-1]    
    O_TEMP = ''.join(['#', '^', O_TYPE, '^', O_WORD, '#'])
    object_entity.append(O_TEMP)
    
    sentence.append(k.replace(S_WORD, S_TEMP).replace(O_WORD, O_TEMP))
    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset


def data_aug(dataset):
  dataset = dataset
  for s,i,j,k in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity'], dataset['label']):
    S_TYPE = i[1:-1].split(", '")[-1].split(': ')[1][1:-1]
    O_TYPE = j[1:-1].split(", '")[-1].split(': ')[1][1:-1]
    if k == "org:members" and S_TYPE == "ORG" and O_TYPE == "ORG":
      dataset.loc[dataset.shape[0]] = [len(dataset), s , j, i, "org:member_of", " "] # id,sentence,subject_entity,object_entity,label,source
    elif k == "org:member_of" and S_TYPE == "ORG" and O_TYPE == "ORG":
      dataset.loc[dataset.shape[0]] = [len(dataset), s , j, i, "org:members", " "]
    elif k == "per:product" and S_TYPE == "PER" and O_TYPE == "ORG":
      dataset.loc[dataset.shape[0]] = [len(dataset), s , j, i, "org:founded_by", " "]
    elif k == "org:founded_by" and S_TYPE == "ORG" and O_TYPE == "PER":
      dataset.loc[dataset.shape[0]] = [len(dataset), s , j, i, "per:product", " "]
    elif k == "per:children" and S_TYPE == "PER" and O_TYPE == "PER":
      dataset.loc[dataset.shape[0]] = [len(dataset), s , j, i, "per:parents", " "]
    elif k == "per:parents" and S_TYPE == "PER" and O_TYPE == "PER":
      dataset.loc[dataset.shape[0]] = [len(dataset), s , j, i, "per:children", " "]
    elif (k == "org:alternate_names" and S_TYPE == "ORG" and O_TYPE == "ORG") or (k == "per:alternate_names" and S_TYPE == "PER" and O_TYPE == "PER") or (k == "per:other_family" and S_TYPE == "PER" and O_TYPE == "PER") or (k == "per:colleagues" and S_TYPE == "PER" and O_TYPE == "PER") or (k == "per:siblings" and S_TYPE == "PER" and O_TYPE == "PER") or (k == "per:spouse" and S_TYPE == "PER" and O_TYPE == "PER") :
      dataset.loc[dataset.shape[0]] = [len(dataset), s , j, i, k, " "]
  return dataset
          

def load_data(dataset_dir, mode="test"):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  if mode == "train":
    pd_dataset = data_aug(pd_dataset)
    # sentence, subject_entity, object_entity, label 전부 동일한 것들 제거
    pd_dataset.drop_duplicates(['sentence', 'subject_entity', 'object_entity', 'label'])
  dataset = preprocessing_dataset_with_punc_sentence12(pd_dataset)
  # if mode == "train": # train data load 시 train_set, val_set 반환
  #   dataset['label_num'] = dataset['label'].map(lambda x: dict_label_to_num(x))
  #   train_set, val_set = train_test_split(dataset, test_size=0.2,
  #                                         shuffle=True, stratify=dataset['label_num'])
  #   return train_set, val_set
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = 'from' + e01 + 'to' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128,
      add_special_tokens=True,
      )
  return tokenized_sentences
