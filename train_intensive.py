import os
import warnings
warnings.filterwarnings("ignore")
import pickle
from itertools import combinations, permutations
from tqdm import tqdm
from callback.progressbar import ProgressBar
import random
import numpy as np
import spacy
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel
from torch.optim import AdamW
from utils import *
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

SPECIAL_TOKENS = True
MASK_FILL = 1e4
MAX_NP_LEN = 5
MAX_CONTEXT_LEN = 80
MAX_LEN = MAX_NP_LEN * 2 + MAX_CONTEXT_LEN + 4
QUERY_LEN = MAX_NP_LEN * 2 + 2

tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, add_special_tokens = False, local_files_only = True,)
if SPECIAL_TOKENS:
    # additional_special_tokens = ['[#S]', '[#O]', '[#Context]'])
    s_id = tokenizer.vocab['[#S]']
    o_id = tokenizer.vocab['[#O]']
    context_id = tokenizer.vocab['[#Context]']

class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)
    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, start_emb_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.start_emb_size = start_emb_size
        self.dense_0 = nn.Linear(hidden_size + start_emb_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions.repeat(1,1,self.start_emb_size)], dim=-1))
        # print(hidden_states.shape, start_positions.shape, x.shape)
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x

class IntensiveModule(BertPreTrainedModel):
    def __init__(self, config,):
        super(IntensiveModule, self).__init__(config)
        self.debug = True
        self.num_labels = config.num_labels
        # self.q_len = QUERY_LEN
        self.c_len = MAX_CONTEXT_LEN + 1
        self.use_attention = True
        self.num_heads = 12
        self.start_emb_size = 1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.atten = nn.MultiheadAttention(config.hidden_size, self.num_heads, dropout=config.hidden_dropout_prob) # 
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        self.end_fc = PoolerEndLogits(config.hidden_size, self.start_emb_size, self.num_labels)
        self.init_weights()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None, context_lens=None):
        # get embedding
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        target_mask = attention_mask[:,:self.c_len] # [bs, L]
        source_mask = attention_mask[:,self.c_len:] # [bs, S]
        sequence_output = outputs[0] # [bs, L+S, E]
        if self.use_attention:
            # cross attention
            Q, K = [x.transpose(1, 0) for x in (sequence_output[:,:self.c_len,:], sequence_output[:,self.c_len:,:])]
            V = K
            sequence_output, _ = self.atten(Q, K, V, ) # key_padding_mask=~source_mask.bool()
            sequence_output = sequence_output.transpose(1, 0)
        else:
            sequence_output = self.dropout(sequence_output)[:,self.q_len:,:]
        # extract phrase
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = torch.softmax(start_logits, -1)
            label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        # print(sequence_output.shape, label_logits.shape)
        end_logits = self.end_fc(sequence_output, label_logits)
        # fill PAD token with mask
        for i in range(len(context_lens)):
            context_len = context_lens[i]
            start_logits[i, context_len:, 0] = MASK_FILL
            start_logits[i, context_len:, 1] = -MASK_FILL
            end_logits[i, context_len:, 0] = MASK_FILL
            end_logits[i, context_len:, 1] = -MASK_FILL
        outputs = (start_logits, end_logits,) # + outputs[2:]
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = target_mask.contiguous().view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels) # L * 2
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs

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
    all_context_len = torch.tensor([f['context_len'] for f in features], dtype=torch.long)
    if gold:
        all_start_positions = torch.tensor([f['start_positions'] for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f['end_positions'] for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_start_positions, all_end_positions, all_context_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)
    else:
        # training = False
        dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_context_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)
    return dataloader

def f1_score_span(pred_start, pred_end, truth_start, truth_end):
    """ pred span: []; truth span: {}; metrics: [FP{TP]FN}"""
    f1_score = 0
    for pred1, pred2, true1, true2 in zip(pred_start, pred_end, truth_start, truth_end):
        if pred2 < pred1:
            pred2 = pred1
        pred2+=1
        true2+=1
        # try:
        TP = min(pred2, true2) - max(pred1, true1)
        if TP <= 0:
            continue
        FP = pred2 - pred1 - TP
        FN = true2 - true1 - TP
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score += 2 * (precision * recall) / (precision + recall)
        # except ZeroDivisionError:
        #     print(pred1, pred2, true1, true2)
        #     raise ValueError('forceful break')
    return f1_score / len(pred_start)

seed_everything()

if __name__ == "__main__":
    with open("./data/cached_features1.pkl", "rb") as f:
        features = pickle.load(f)
    train_size = int(len(features) * 0.8)
    dev_size = len(features) - train_size
    train_features = features[:train_size]
    dev_featuers = features[train_size:]
    train_dataloader = load_data(train_features, True)
    dev_dataloader = load_data(dev_featuers, False)

    config = BertConfig.from_pretrained("./bert-base-uncased",num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    model = IntensiveModule.from_pretrained("./bert-base-uncased", config=config).cuda()

    # initialize optimizer and scheduler 
    learning_rate = 5e-5
    weight_decay = 0.01
    crf_learning_rate = 5e-5
    epochs = 10
    patience = 2
    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    num_train_steps = len(train_dataloader) * epochs
    warmup_steps = num_train_steps / (2 * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

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
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "start_positions": batch[3], "end_positions": batch[4], 'context_lens': batch[-1]}
            optimizer.zero_grad()
            with autocast():
                loss, start_scores, end_scores = model(**inputs)
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar(step, {'batch_avg_loss': loss.item() / batch[0].shape[0]})
        train_loss /= train_size
        print('training loss of epoch %d: %f'%(epoch + 1, train_loss))
        # raise(ValueError('break'))
        pred_start = []
        pred_end = []
        truth_start = []
        truth_end = []
        print("validating...")
        with torch.no_grad():
            model.eval()
            dev_loss = 0
            for batch in dev_dataloader:
                batch = [f.cuda() for f in batch]
                inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "start_positions": batch[3], "end_positions": batch[4], 'context_lens': batch[-1]}
                with autocast():
                    loss, start_scores, end_scores = model(**inputs) # score.shape = bs*max_len*2
                dev_loss += loss.item()
                start_scores = torch.softmax(start_scores,dim=2)[:,:,1].view(start_scores.shape[0], start_scores.shape[1])
                end_scores = torch.softmax(end_scores,dim=2)[:,:,1].view(end_scores.shape[0], end_scores.shape[1])
                batch_pred_start = torch.argmax(start_scores, dim=1) # bs*1
                batch_pred_end = torch.argmax(end_scores, dim=1) # bs*1
                batch_pred_start = batch_pred_start.flatten().detach().cpu().numpy().tolist()
                batch_pred_end = batch_pred_end.flatten().detach().cpu().numpy().tolist()
                pred_start.extend(batch_pred_start)
                pred_end.extend(batch_pred_end)
                batch_truth_start = torch.argmax(batch[3], dim=1).flatten().detach().cpu().numpy().tolist()
                batch_truth_end = torch.argmax(batch[4], dim=1).flatten().detach().cpu().numpy().tolist()
                truth_start.extend(batch_truth_start)
                truth_end.extend(batch_truth_end)
        f1 = f1_score_span(pred_start, pred_end, truth_start, truth_end)
        torch.cuda.empty_cache()
        dev_loss /= dev_size
        print('dev_loss:', dev_loss)
        if f1 >= best_f1:
            print('validation f1_score increased from %f to %f, save model...'%(best_f1, f1))
            torch.save(model.state_dict(), 'IntensiveModule.bin')
            best_f1 = f1
        else:
            print('validation f1_score did not increase; got %f, but the best is %f'%(f1, best_f1))
            if epoch >= 5:
                patience -= 1
        if patience <= 0:
            print('model overfitted, call early stop')
            break