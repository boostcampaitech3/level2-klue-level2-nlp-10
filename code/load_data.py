import pickle as pickle
import os
import pandas as pd
from df_edit import better_df
from preprocess import Preprocess
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

# def correction(idx, S_TYPE, O_TYPE, LABEL):
  # """ (Deprecated) 데이터 전수조사를 통해 찾은 오류 수정함수 (현재 사용하지 않기로 합의함.)

  # Returns:
  #     _type_: _description_
  # """
#   A = [4876] # "POH" / - / "no_relation"
#   B = [1842] # - / "PER" / "no_relation"
#   C = [15567, 22566, 24813, 1212, 7539, 25598, 
#       26958, 2090, 5522, 7649, 29766, 502] # - / - / "no_relation"
#   D = [4127, 1539, 3079, 16095, 23884] # - / - / "org:alternate_names"
#   E = [8177, 24560] # - / - / "per:parents"
#   F = [1413, 1550, 13184, 15970] # - / "LOC" / -
#   G = [125, 392, 656, 959, 1129, 1208, 2020, 2226, 2419, 3010, 
#       3620, 4584, 4665, 6001] # "ORG" / - / "org:top_members/employees"
#   H = [1235, 4845, 21082, 23490, 6061] # - / - / "per:title"
#   I = [18305] # - / - / "per:employee_of"
#   J = [28891, 4662] # "ORG" / - / -
#   K = [19518] # "PER" / - / "per:place_of_residence"
#   L = [19531, 29357] # - / - / "per:place_of_death"
#   M = [4077, 4313, 6291, 12110, 23596, 30324, 30398, 32246] # - / "PER" / -
#   N = [16238, 23328, 30202] # - / - / "org:top_members/employees"
#   O = [1990, 592, 2171, 3539, 8745, 11780, 22895, 31401, 9055] # - / - / "org:product"
#   P = [16422] # - / - / "org:political/religious_affiliation"
#   Q = [4958, 18772, 31605, 1469, 3485, 11346, 12351, 19177, 19489, 30650, 
#       14043] # - / - / "org:member_of"
#   R = [21482] # - / - / "per:origin"
#   S = [29316, 30018] # - / "ORG" / "org:alternate_names"
#   T = [7365, 14859, 19704, 25507, 26034] # - / "ORG" / "org:member_of"
#   U = [3118] # "ORG" / "ORG" / -
#   V = [14724, 16063, 16613, 20597, 21196, 26294, 
#       27315, 27504, 29716, 29940, 32351] # - / "POH" / org:product"
#   W = [13, 364, 588, 1216, 4212, 5061, 6090, 8604, 
#       11376, 14836, 15630, 18157, 18582, 21695, 23361, 
#       23365, 23785, 25992, 29353, 31302, 31661, 32218, 
#       1588, 2367, 2480, 3381, 3486, 5256, 5415, 5535, 5869, 6321, 7430, 
#       7535, 7582, 5974, 17805, 29629] # - / "ORG" / -
#   Y = [286, 3226, 4423, 6281, 7441, 10309, 11401, 12092, 12309,
#       12714, 13327, 13617, 13502, 16593, 18065, 18502, 18558, 18848, 20268, 
#       21989, 22478, 22676, 2305923656, 25227, 25576, 25703, 23819, 22940, 
#       21565, 21255, 21340, 20763, 20344, 20340, 18812, 17352, 17048, 16946, 
#       16726, 16246, 15652, 15190, 13118, 12717, 11810, 11353, 11354, 11360, 
#       10628, 10538, 10351, 10280, 9938, 9761, 9356, 9255, 8290, 7294, 6903, 
#       6617, 6186, 5989, 5628, 3848, 3031, 2545, 1307, 1093, 806, 135, 25930, 
#       26045, 26478, 26532, 26945, 27148, 27683, 27721, 27745, 27795, 28330, 
#       29143, 29375, 29382, 30235, 30910, 31210, 31234, 31348, 31865, 31901, 
#       32409, 32422] # - / "LOC" / "org:place_of_headquarters"
#   Z = [690, 1292, 1427, 2228, 2359, 3187, 3213, 3319, 3736, 4016, 4470, 
#       4593, 5226, 5384, 5472, 6381, 6531, 6649, 6976, 6987, 7549] # "ORG" / - / "org:member_of"
#   A_ = [12478, 13895, 14161, 22443, 30359] # - / "POH" / -
#   B_ = [575, 7445, 9461, 10230, 10491] # - / - / "per:alternate_names"
#   C_ = [11537, 12773, 13559] # "PER" / - / "per:alternate_names"
#   D_ = [3033, 27954] # - / "DAT" / -
#   E_ = [8297, 9233, 24175] # - / - / "org:members"
#   F_ = [14766, 17462, 24482] # - / - / "per:employee_of"

#   if idx in A: S_TYPE, LABEL = "POH", "no_relation"
#   elif idx in B: O_TYPE, LABEL = "PER", "no_relation"
#   elif idx in C: LABEL = "no_relation"
#   elif idx in D: LABEL = "org:alternate_names"
#   elif idx in E: LABEL = "per:parents"
#   elif idx in F: O_TYPE = "LOC"
#   elif idx in G: S_TYPE, LABEL = "ORG", "org:top_members/employees"
#   elif idx in H: LABEL = "per:title"
#   elif idx in I: LABEL = "per:employee_of"
#   elif idx in J: S_TYPE = "ORG"
#   elif idx in K: S_TYPE, LABEL = "PER", "per:place_of_residence"
#   elif idx in L: LABEL = "per:place_of_death"
#   elif idx in M: O_TYPE = "PER"
#   elif idx in N: LABEL = "org:top_members/employees"
#   elif idx in O: LABEL = "org:product"
#   elif idx in P: LABEL = "org:political/religious_affiliation"
#   elif idx in Q: LABEL = "org:member_of"
#   elif idx in R: LABEL = "per:origin"
#   elif idx in S: O_TYPE, LABEL = "ORG", "org:alternate_names"
#   elif idx in T: O_TYPE, LABEL = "ORG", "org:member_of"
#   elif idx in U: S_TYPE, O_TYPE = "ORG", "ORG"
#   elif idx in V: O_TYPE, LABEL = "POH", "org:product"
#   elif idx in W: O_TYPE = "ORG"
#   elif idx in Y: O_TYPE, LABEL = "LOC", "org:place_of_headquarters"
#   elif idx in Z: S_TYPE, LABEL = "ORG", "org:member_of"
#   elif idx in A_: O_TYPE = "POH"
#   elif idx in B_: LABEL = "per:alternate_names"
#   elif idx in C_: S_TYPE, LABEL = "PER", "per:alternate_names"
#   elif idx in D_: O_TYPE = "DAT"
#   elif idx in E_: LABEL = "org:members"
#   elif idx in F_: LABEL = "per:employee_of"

#   return S_TYPE, O_TYPE, LABEL

# Typed entity marker(punct) to Query and Sentence

def preprocessing_dataset_with_sentence(dataset : pd.DataFrame):
  """ Sentence 및 subject/object entity에 Typed entity marker를 추가합니다.

  Args:
      dataset (DataFrame): 전처리 및 개선 작업 완료한 DataFrame

  Returns:
      DataFrame: Typed entity marker를 추가한 DataFrame
  """  
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
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