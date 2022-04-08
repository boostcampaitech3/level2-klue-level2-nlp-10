from turtle import forward
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from transformers import AutoModel
import torch
import torch.nn as nn

import torch.nn.functional as F

class ModelStatic(nn.Module):
  def __init__(self, MODEL_NAME,device,batchs_per_epoch):
      super().__init__()
      print(MODEL_NAME, 'model loading..')
      self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
      self.model_config.num_labels = 30
      self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config = self.model_config)
      self.static_metrics = self.init_static_metrics(self.model_config.num_labels)
      self.static_cnt = 0
      self.device = device
      self.batchs_per_epoch = batchs_per_epoch
      # self.fc = nn.Linear(self.model_config.num_labels*(self.model_config.num_labels+1),self.model_config.num_labels)
      # self.relu = nn.ReLU()
      print('*** Model_Static initialized..!!')

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, 
              position_ids=None, head_mask=None):
      logits = self.model(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                          attention_mask=attention_mask, head_mask=head_mask).logits
      # print("logits.shape:",logits.shape)
      # ref_p = self.static_cnt/self.batchs_per_epoch
      ref_p = 1
      # if self.static_cnt >= self.batchs_per_epoch//2:
      #   self.static_cnt = 0
      #   self.static_metrics = self.init_static_metrics(self.model_config.num_labels)
      
      t_list = []
      for i in range(len(logits)):
        # t_list.append(torch.cat([F.softmax(logits[i],dim=0).to(self.device),
        #                         F.softmax(torch.flatten(self.static_metrics)).to(self.device)],dim=0))
        # s_m = torch.zeros_like(logits[i])
        pred_label = torch.argmax(logits[i],dim=0)
        # p_out = 1-F.softmax(self.static_metrics[pred_label],dim=0)[pred_label]
        # if p_out > 0.1:
        #   s_m[pred_label] = (p_out)*ref_p
        t_list.append(F.softmax(self.static_metrics[pred_label],dim=0))
      static_p = torch.stack(t_list)
      # logits = self.fc(static_p.to(self.device))
      logits = F.softmax(logits,dim=0).to(self.device) + static_p.to(self.device)
      # static_p = torch.stack([F.softmax(self.static_metrics[torch.argmax(logit,dim=0)],dim=0)*ref_p for logit in logits])
      # logits = logits + self.relu(logits - static_p.to(self.device))
          # m_logits = F.softmax(logits[i],dim=0).to(device) + F.softmax(self.staic_metrics[torch.argmax(logits[i],dim=0)],dim=0).to(device)/10
          # m_logits.to(device)
          # logits[i] = m_logits
      self.static_cnt += 1
      # if self.static_cnt % 50 == 0:
      #   self.static_metrics = self.init_static_metrics(self.model_config.num_labels)
    
      return {'logits' : logits}

  def init_static_metrics(self,num_labels):
      return torch.zeros([num_labels, num_labels], dtype=torch.float)
  
  def update_static_metrics(self,logits,labels):
      for logit, label in zip(logits,labels):
          self.static_metrics[torch.argmax(logit)][label] += 1
      return 0

# BiLSTM
class Static_Model_BiLSTM(nn.Module):
  def __init__(self, MODEL_NAME, NUM_LAYERS , device, batchs_per_epoch):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= NUM_LAYERS, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
    self.static_metrics = self.init_static_metrics(self.model_config.num_labels)
    self.static_cnt = 0
    self.device = device
    self.batchs_per_epoch = batchs_per_epoch
    self.relu = nn.ReLU()
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)

    hidden, (last_hidden, last_cell) = self.lstm(output)
    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # hidden : (batch, max_len, hidden_dim * 2)
    # last_hidden : (2, batch, hidden_dim)
    # last_cell : (2, batch, hidden_dim)
    # output : (batch, hidden_dim * 2)

    logits = self.fc(output)
    # (batch, num_labels)

    ref_p = self.static_cnt/self.batchs_per_epoch
    if self.static_cnt >= self.batchs_per_epoch:
        self.static_cnt = 0
        self.static_metrics = self.init_static_metrics(self.model_config.num_labels)

    t_list = []
    for i in range(len(logits)):
      pred_label = torch.argmax(logits[i],dim=0)
      t_list.append(F.softmax(self.static_metrics[pred_label],dim=0)*ref_p)
    static_p = torch.stack(t_list)
    logits = logits + self.relu(logits - static_p.to(self.device))
    self.static_cnt += 1

    return {'logits' : logits}
  
  def init_static_metrics(self,num_labels):
      return torch.zeros([num_labels, num_labels], dtype=torch.float)
  
  def update_static_metrics(self,logits,labels):
      for logit, label in zip(logits,labels):
          self.static_metrics[torch.argmax(logit)][label] += 1
      return 0

# BiLSTM
class Model_BiLSTM(nn.Module):
  def __init__(self, MODEL_NAME,NUM_LAYERS=1):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= NUM_LAYERS, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)

    hidden, (last_hidden, last_cell) = self.lstm(output)
    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # hidden : (batch, max_len, hidden_dim * 2)
    # last_hidden : (2, batch, hidden_dim)
    # last_cell : (2, batch, hidden_dim)
    # output : (batch, hidden_dim * 2)

    logits = self.fc(output)
    # (batch, num_labels)

    return {'logits' : logits}

# BiGRU
class Model_BiGRU(nn.Module):
  def __init__(self, MODEL_NAME,NUM_LAYERS=1):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.gru= nn.GRU(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= NUM_LAYERS, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
    print('*** Model_BiGRU init....')
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)

    hidden, last_hidden = self.gru(output)
    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # hidden : (batch, max_len, hidden_dim * 2)
    # last_hidden : (2, batch, hidden_dim)
    # output : (batch, hidden_dim * 2)

    logits = self.fc(output)
    # logits : (batch, num_labels)

    return {'logits' : logits}

# FC
class Model_FC(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.fc = nn.Linear(self.hidden_dim * 128, self.model_config.num_labels)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (batch, max_len, hidden_dim)
    output = output.view(output.shape[0], -1) # (batch, max_len * hidden_dim)
    output = self.fc(output) # (batch, num_labels)
    output = self.relu(output)
    logits = self.softmax(output)
    
    return {'logits' : logits}