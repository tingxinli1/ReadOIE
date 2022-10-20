import os
import warnings
warnings.filterwarnings("ignore")

from callback.progressbar import ProgressBar
import random
import numpy as np
import pickle
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
# from crf import CRF
from transformers import AutoTokenizer, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel
from torch.optim import AdamW
from utils import *
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss

SPECIAL_TOKENS = True

class ExtensiveModule(BertPreTrainedModel):
    def __init__(self, config,):
        super(ExtensiveModule, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        if labels is not None:
            loss_fn = CrossEntropyLoss()
            batch_loss = loss_fn(logits, labels)
            return batch_loss, logits
        else:
            return logits

if SPECIAL_TOKENS:
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, add_special_tokens = False, local_files_only = True,)
    # additional_special_tokens = ['[#S]', '[#O]', '[#Context]'])
    s_id = tokenizer.vocab['[#S]']
    o_id = tokenizer.vocab['[#O]']
    context_id = tokenizer.vocab['[#Context]']
else:
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, add_special_tokens = False, local_files_only = True)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def load_data(features, training, batch_size=32, gold=True):
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    if gold:
        all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)
    else:
        # training = False
        dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)
    return dataloader

if __name__ == "__main__":
    seed_everything()
    num_samples = int(2e4)
    train_size = int(num_samples * 0.8)
    dev_size = num_samples - train_size
    with open("./data/cached_features2.pkl", "rb") as f:
        train_features, dev_featuers = pickle.load(f)
    train_dataloader = load_data(train_features, True)
    dev_dataloader = load_data(dev_featuers, False)
    train_size, dev_size = len(train_dataloader), len(dev_dataloader)

    # train model
    config = BertConfig.from_pretrained("./bert-base-uncased",num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    model = ExtensiveModule.from_pretrained("./bert-base-uncased", config=config).cuda()
    # initialize optimizer and scheduler
    learning_rate = 5e-5
    weight_decay = 0.01
    epochs = 6
    patience = 1
    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    num_train_steps = len(train_dataloader) * epochs
    warmup_steps = num_train_steps / (2 * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    # train loops
    # best_loss = float('inf')
    best_f1 = 0
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=epochs)
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch+1)
        for step, batch in enumerate(train_dataloader):
            batch = [f.cuda() for f in batch]
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "labels": batch[3]}
            optimizer.zero_grad()
            with autocast():
                loss, logits = model(**inputs)
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar(step, {'batch_avg_loss': loss.item() / batch[0].shape[0]})
        train_loss /= train_size
        print('training loss of epoch %d: %f'%(epoch + 1, train_loss))
        pred_labels = []
        true_labels = []
        print("validating...")
        with torch.no_grad():
            model.eval()
            dev_loss = 0
            for batch in dev_dataloader:
                batch = [f.cuda() for f in batch]
                inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "labels": batch[3]}
                with autocast():
                    loss, logits = model(**inputs) # score.shape = bs*max_len*2
                dev_loss += loss.item()
                batch_pred_labels = torch.argmax(logits,dim=1).flatten().detach().cpu().numpy().tolist()
                batch_true_labels = inputs['labels'].flatten().detach().cpu().numpy().tolist()
                pred_labels.extend(batch_pred_labels)
                true_labels.extend(batch_true_labels)
        f1 = f1_score(true_labels, pred_labels, average='binary')
        torch.cuda.empty_cache()
        dev_loss /= dev_size
        print('dev_loss:', dev_loss)
        if f1 >= best_f1:
            print('validation f1_score increased from %f to %f, save model...'%(best_f1, f1))
            torch.save(model.state_dict(), 'ExtensiveModule.bin')
            best_f1 = f1
        else:
            print('validation f1_score did not increase; got %f, but the best is %f'%(f1, best_f1))
            patience -= 1
        if patience <= 0:
            print('model overfitted, call early stop')
            break