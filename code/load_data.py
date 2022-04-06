import pickle as pickle
import os
import pandas as pd
from df_edit import better_df
from preprocess import Preprocess
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

# Typed entity marker(punct) to Subject/Object Entity and Sentence
def preprocessing_dataset_with_sentence(dataset : pd.DataFrame):
  """ Sentence 및 Subject/Object Entity에 Typed entity marker를 추가합니다.

  Args:
      dataset (DataFrame): 전처리 및 개선 작업 완료한 DataFrame

  Returns:
      DataFrame: Typed entity marker를 추가한 DataFrame
  """  
  subject_entity = []
  object_entity = []
  sentence = []
  for S_WORD,S_TYPE,O_WORD,O_TYPE,SEN in zip(dataset['subj_word'], dataset['subj_type'], dataset['obj_word'], dataset['obj_type'], dataset['sentence']): 
    
    S_TEMP = ' '.join(['@', '*', '['+S_TYPE+']', '*', S_WORD, '@'])
    # S_TEMP = ' '.join(['@', '+', S_TYPE, '+', S_WORD, '@'])
    subject_entity.append(S_TEMP)
  
    O_TEMP = ' '.join(['#', '^', '['+O_TYPE+']', '^', O_WORD, '#'])
    # O_TEMP = ' '.join(['#', '^', O_TYPE, '^', O_WORD, '#'])
    object_entity.append(O_TEMP)

    sentence.append(SEN.replace(S_WORD, S_TEMP).replace(O_WORD, O_TEMP))

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence, 'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir : str):
  """
  Train 용 dataframe 생성 함수
  1) csv 파일을 경로에 맞게 불러 옵니다. 
  2) dataframe을 개선하고 전처리 과정을 수행합니다.
  3) type + 특수문자를 Sentence에 추가합니다.

  Args:
      dataset_dir (file_dir): train data의 위치 (default : "/opt/ml/dataset/train/train.csv")

  Returns:
      type: Pandas Dataframe
  """  

  # 주어진 경로에서 csv 파일을 불러와 dataframe으로 읽습니다.
  pd_dataset = pd.read_csv(dataset_dir)

  # dataframe 개선을 합니다. word, index, type을 풀어 각각 하나의 column으로 담습니다. label num을 포함합니다.
  pd_dataset = better_df(pd_dataset,0)
  
  # 모든 값이 일치한 data를 삭제합니다. 또한 label만 다른 data 쌍에 대해 합의를 통해 삭제할 대상을 선정하여 삭제합니다.
  dataset = Preprocess(pd_dataset)

  # 특수 문자+ type을 subj, obj entity + 문장에 추가합니다.
  # 논문 An Improved Baseline for Sentence-level Relation Extraction (2021) 참고
  dataset = preprocessing_dataset_with_sentence(dataset)

  return dataset


def load_data_test(dataset_dir : str):
  """
  inference용 dataframe 생성 함수
  1) csv 파일을 경로에 맞게 불러 옵니다. 
  2) inference 할 때는 전처리 및 dataframe 개선을 하지 않습니다.
  3) type + 특수문자를 Sentence에 추가합니다.

  Returns:
      type : Pandas Dataframe
  """  
  # inference 할 때는 전처리 및 dataframe 개선을 하지 않습니다.
  pd_dataset = pd.read_csv(dataset_dir)

  # dataframe 개선을 합니다. word, index, type을 풀어 각각 하나의 column으로 담습니다. 
  pd_dataset = better_df(pd_dataset,1)
  
  # 특수 문자+ type을 subj, obj entity + 문장에 추가합니다.
  # 논문 An Improved Baseline for Sentence-level Relation Extraction (2021) 참고
  dataset = preprocessing_dataset_with_sentence(pd_dataset)

  return dataset


def tokenized_dataset(dataset : pd.DataFrame, tokenizer):
  """ 
  Query를 기존 BERT 학습 방법과 동일하게 한 문장으로 구성합니다. 주어진 tokenizer에 따라 sentence를 tokenizing 합니다.
  max_length = 256으로 설정되어 있습니다. 본인이 필요에 따라 바꾸시면 됩니다.

  Args:
      dataset (DataFrame): 개선, 전처리, 문장에 type entity marker를 추가한 dataset
      tokenizer (AutoTokenizer): 주어진 tokenizer

  Returns:
      list: 토큰화된 문장
  """  
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = e01 + ' 과 ' + e02 + '의 관계'
    concat_entity.append(temp)
  
  tokenized_sentence = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids = False
      )
  
  return tokenized_sentence