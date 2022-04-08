import pickle as pickle
import os
import pandas as pd
import torch


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

# Typed Entity Marker(Punct) to Only Query
def TEMP_preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  remove_idxs = []
  relabel_idxs = []
  relabel_labels = []
  retype_idxs = []
  retype_entitys = []

  # traning에서만 데이터가 3만개가 넘으므로 inference에서는 필터링이 되지않음.
  if len(dataset) > 30000:
    # 중복(삭제대상) 데이터 목록
    # 17142??
    remove_idxs = [3547,3296,10202,27920,22222,7168,15776,19571,25368,10616,
                  25094,20898,18171,27325,22772,8693,12829,14658,31786,24788,
                  8364,29180,31896,10043,22090,32282,14094,22258,31785,28350,
                  21757,31510,29511,20062,27116,31038,26044,22641,24373,30640,
                  28608,29854,28730,28010,29674,30378,32274,
                  18458,27755,20838,7080,25673]

    # 라벨 수정 데이터 목록
    # 17142 ??
    relabel_idxs = [6749,7276,13371]
    relabel_labels = ['org:top_members/employees','per:employee_of',
                      'per:employee_of']

    # type 수정 데이터 목록
    # 11554 ??
    # PER: 사람이름/ LOC: 지명 / ORG: 기관명 / POH: 기타 / DAT: 날짜
    # TIM: 시간 / DUR: 기간 / MNY: 통화 / PNT: 비율 / NOH: 기타 수량표현
    retype_idxs = [2464,30258,6530,7264,15128,11554,28644,28281]
    retype_entitys = [['obj_entity','ORG'],['obj_entity','POH'],['obj_entity','PER'],
                      ['obj_entity','ORG'],['obj_entity','POH'],['obj_entity','LOC'],
                      ['obj_entity','PER'],['sub_entity','ORG']]

  ids = []
  sentences = []
  subject_entity = []
  object_entity = []
  labels = []
  for id_, sentence, sub_ent, obj_ent, label in zip(dataset['id'],
                                                    dataset['sentence'],
                                                    dataset['subject_entity'],
                                                    dataset['object_entity'],
                                                    dataset['label']):
    # 중복 데이터 삭제
    if id_ in remove_idxs:
      continue
    ids.append(id_)
    sentences.append(sentence)

    S_WORD = eval(sub_ent)["word"]
    S_TYPE = eval(sub_ent)["type"]
    S_TEMP = ' '.join(['@', '*','[', S_TYPE, ']','*', S_WORD, '@'])
    subject_entity.append(S_TEMP)
    
    O_WORD = eval(obj_ent)["word"]
    O_TYPE = eval(obj_ent)["type"]    
    O_TEMP = ' '.join(['#', '^','[', O_TYPE, ']','^', O_WORD, '#'])
    object_entity.append(O_TEMP)

    # 타입수정
    if id_ in retype_idxs:
      entity = retype_entitys[retype_idxs.index(id_)]
      if entity[0] == 'sub_entity':
        subject_entity[-1] = entity[1]
      else:
        object_entity[-1] = entity[1]

    labels.append(label)
    # 라벨수정
    if id_ in relabel_idxs:
      labels[-1] = relabel_labels[relabel_idxs.index(id_)]

  out_dataset = pd.DataFrame({'id':ids, 'sentence':sentences,'subject_entity':subject_entity,'object_entity':object_entity,'label':labels,})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  # dataset = preprocessing_dataset(pd_dataset)
  dataset = TEMP_preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = e01 + '과' + e02 + '의 관계'
    concat_entity.append(temp)

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids = False
      )
  return tokenized_sentences

def TEMP_tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + ' 과 ' + e02 + '의 관계'
    concat_entity.append(temp)
  
  tokenized_sentence = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128,
      add_special_tokens=True,
      return_token_type_ids = False # also called segment IDs
      )
  
  return tokenized_sentence
