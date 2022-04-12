from load_data import *
import pandas as pd
import pickle as pickle
import numpy as np
from train import *

def inference(prob):  
  pred = np.argmax(prob, axis=-1)
  return pred.tolist(), prob.tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_output(output_dir):
  pd_output = pd.read_csv(output_dir)
  probs = []
  for prob in pd_output['probs']:
    probs.append(eval(prob))
  
  return probs

def soft_voting():
  K = 5 # ensemble할 model 개수
  outputs_probs = []
  for K_ in range(1, K + 1):
    output_dir = "./prediction/output (" + str(K_) + ")" + ".csv"
    output_probs = load_output(output_dir)
    outputs_probs.append(output_probs)
  
  samples_probs = []
  for sample in range(len(outputs_probs[0])):
    sample_probs = []
    for label in range(30):
      SUM = 0
      for K_ in range(K):
        SUM += outputs_probs[K_][sample][label]
      AVG = SUM / K
      sample_probs.append(AVG)
    samples_probs.append(sample_probs)

  output_prob = torch.FloatTensor(samples_probs)
  pred_answer, output_prob = inference(output_prob)
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  output = pd.DataFrame({'id':[sample for sample in range(len(outputs_probs[0]))],'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/softvoting.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  print('---- Finish! ----')

if __name__ == '__main__':
  soft_voting()