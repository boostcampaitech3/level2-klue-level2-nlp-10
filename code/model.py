from turtle import forward
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from transformers import AutoModel
import torch
import torch.nn as nn

class Basic(nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()
        print(MODEL_NAME, 'model loading..')
        self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
        self.model_config.num_labels = 30
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config = self.model_config)
    def forward(self,input):
        return self.model(input)

# BiLSTM
class Model_BiLSTM(nn.Module):
  def __init__(self, MODEL_NAME,NUM_LAYERS=2):
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