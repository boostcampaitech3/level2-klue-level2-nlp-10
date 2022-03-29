import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback
from load_data import *
import wandb
from loss import *
import random
from sklearn.model_selection import StratifiedKFold

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# BiLSTM
class Model_BiLSTM(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 1, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (batch, max_len, hidden_dim)

    hidden, (last_hidden, last_cell) = self.lstm(output)
    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # hidden : (batch, max_len, hidden_dim * 2)
    # last_hidden : (2, batch, hidden_dim)
    # last_cell : (2, batch, hidden_dim)
    # output : (batch, hidden_dim * 2)

    logits = self.fc(output) # (batch, num_labels)

    return {'logits' : logits}

# BiGRU
class Model_BiGRU(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.gru= nn.GRU(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 1, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
  
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

class Custom_Trainer(Trainer) :
    def __init__(self,loss_name, *args, **kwargs) :
        super().__init__(*args,**kwargs)
        self.loss_name = loss_name
    
    def compute_loss(self, model, inputs, return_outputs = False) :
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')

        # 인덱스에 맞춰서 과거 ouput을 다 저장
        if self.args.past_index >=0:
            self._past= outputs[self.args.past_index]

        if self.loss_name == 'CrossEntropyLoss':
            custom_loss= torch.nn.CrossEntropyLoss().to(device)
            loss= custom_loss(outputs['logits'], labels)
        
        elif self.loss_name == 'FOCAL_LOSS':
            custom_loss = FocalLoss(gamma = 1).to(device)
            loss = custom_loss(outputs['logits'],labels)
        return (loss, outputs) if return_outputs else loss


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  wandb.log({'micro f1 score': f1})
  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  #MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/dataset/train/train.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  # train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)
  split_dataset = train_dataset
  split_label = train_dataset['label'].values

  # tokenizing dataset
  # tokenized_train = TEMP_tokenized_dataset(train_dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
  kfold = StratifiedKFold(n_splits = 10, random_state = seed_everything(42))
  for fold,(train_idx,val_idx) in enumerate(kfold.split(split_dataset, split_label)) :
    wandb_run = wandb.init(project = 'huggingface', name = f'KFOLD_{fold}_TEM_with focal_loss')
    print("-"*20,f'fold: {fold} start',"-"*20)
    train_dataset = split_dataset.iloc[train_idx]
    val_dataset = split_dataset.iloc[val_idx]

    train_label = label_to_num(train_dataset['label'].values)
    val_label = label_to_num(val_dataset['label'].values)

    tokenized_train= TEMP_tokenized_dataset(train_dataset, tokenizer)
    tokenized_val= TEMP_tokenized_dataset(val_dataset, tokenizer)

    trainset= RE_Dataset(tokenized_train, train_label)
    valset= RE_Dataset(tokenized_val, val_label)

    model =  Model_BiGRU(MODEL_NAME)
    model.to(device)
    save_dir = f'./results/KFOLD_{fold}_TEM_with focal_loss'

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_strategy = 'epoch',
        save_total_limit=1,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=5,              # total number of training epochs
        learning_rate=3e-5,               # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        # warmup_steps=500,                # number of warmup steps for learning rate scheduler
        warmup_ratio = 0.1,
        weight_decay=0.01,               # strength of weight decay
        # label_smoothing_factor=0.1,
        # lr_scheduler_type = 'constant_with_warmup',
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        load_best_model_at_end = True,
        report_to = 'wandb'
        )

    trainer = Custom_Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=trainset,         # training dataset
        eval_dataset=valset,             # evaluation dataset
        compute_metrics=compute_metrics,      # define metrics function
        loss_name = 'CrossEntropyLoss', # CE
        callbacks = [EarlyStoppingCallback(early_stopping_patience= 3)]
    )

    # train model
    trainer.train()
    if not os.path.exists(f'./best_model/fold_{fold}'):
            os.makedirs(f'./best_model/fold_{fold}')
    torch.save(model.state_dict(), os.path.join(f'./best_model/fold_{fold}', 'pytorch_model.bin'))
    wandb_run.finish()
    print(f'fold_{fold} fin')

def main():
  train()

if __name__ == '__main__':
#   wandb.init(project="huggingface",name = "Typerd entity marker(punct) with focal_loss")
  main()