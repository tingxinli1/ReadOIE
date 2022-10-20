import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from itertools import combinations
from tqdm import tqdm
import random
import numpy as np
import spacy
import torch
from torch.cuda.amp import autocast
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel
from utils import *
from torch.nn import CrossEntropyLoss
from evaluate.evaluate import Benchmark
from evaluate.matcher import Matcher
from evaluate.generalReader import GeneralReader

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

class ExtensiveModule(BertPreTrainedModel):
    def __init__(self, config):
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

def load_data1(features, training, batch_size=32, gold=True):
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    all_context_len = torch.tensor([f['context_len'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)
    return dataloader

def load_data2(features, training, batch_size=32, gold=True):
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)
    return dataloader

def get_phrase(token_ids, rel_start, rel_end):
    rel_token_ids = token_ids[rel_start:rel_end]
    return tokenizer.decode(rel_token_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='web', help='task name, one of web, nyt, or penn', type=str)
    args = parser.parse_args()
    seed_everything()
    # load data
    # max_len = 150
    TASK_NAME = args.task_name
    with open('./data/benchmarks/%s.txt'%TASK_NAME, 'r', encoding='utf-8') as f:
        oie_test = [line.strip() for line in f.readlines()]
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, return_offsets_mapping=True, add_special_tokens = False, local_files_only = True)
    nlp = spacy.load("en_core_web_trf")
    # nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    doc_ids = []
    input_text = []
    test_features1 = []
    test_features2 = []
    print('reading test data...')
    for doc_id, sent in tqdm(list(enumerate(oie_test))):
        text = sent.lstrip().rstrip()
        try:
            sent_spacy = nlp(text)
        except:
            print(text)
            break
        tokens = [token.text for token in sent_spacy]
        pos_tags = [token.pos_ for token in sent_spacy]
        # np_bounds = [[chunk.start, chunk.end] for chunk in sent_spacy.noun_chunks]
        np_bounds = [[ent.start, ent.end] for ent in sent_spacy.ents]
        if len(np_bounds) < 2:
            continue
        # eliminate determiners at the head of the noun phrases
        for i in range(len(np_bounds)):
            np_start, np_end = np_bounds[i]
            tags = pos_tags[np_start:np_end]
            for tag in tags:
                if tag == 'DET':
                    np_start+=1
                else:
                    break
            np_bounds[i] = [np_start, np_end]
        # generate np pairs
        for span_pair in combinations(np_bounds, 2):
            doc_ids.append(doc_id)
            np_span1, np_span2 = sort_span_pair(span_pair)
            # np_span1, np_span2 = span_pair
            np1 = ' '.join(tokens[np_span1[0]: np_span1[1]])
            np2 = ' '.join(tokens[np_span2[0]: np_span2[1]])
            input_text.append(text + '[SEP]' + np1 + '[SEP]' + np2)
            # load input for extracting model
            pad_id = tokenizer.pad_token_id
            context_input_ids = tokenizer(text, add_special_tokens = False, max_length = MAX_CONTEXT_LEN, truncation=True, return_tensors='pt')['input_ids'][0].tolist()
            context_len = len(context_input_ids)
            sub_inputs = tokenizer(np1, add_special_tokens = False, max_length = MAX_NP_LEN, truncation=True, return_tensors='pt')['input_ids'][0].tolist()
            sub_mask = [1] * (len(sub_inputs)) + [0] * (MAX_NP_LEN - len(sub_inputs))
            sub_inputs = sub_inputs + [pad_id] * (MAX_NP_LEN - len(sub_inputs))
            obj_inputs = tokenizer(np2, add_special_tokens = False, max_length = MAX_NP_LEN, truncation=True, return_tensors='pt')['input_ids'][0].tolist()
            obj_mask = [1] * (len(obj_inputs)) + [0] * (MAX_NP_LEN - len(obj_inputs))
            obj_inputs = obj_inputs + [pad_id] * (MAX_NP_LEN - len(obj_inputs))
            context_mask = [1] * len(context_input_ids) + [0] * (MAX_CONTEXT_LEN - len(context_input_ids))
            context_input_ids = context_input_ids + [pad_id] * (MAX_CONTEXT_LEN - len(context_input_ids))
            if SPECIAL_TOKENS:
                input_ids = [tokenizer.cls_token_id] + context_input_ids + [s_id] + sub_inputs + [o_id] + obj_inputs + [tokenizer.sep_token_id] # total max_len: 128+10+1+10+1=150
            else:
                input_ids = [tokenizer.cls_token_id] + context_input_ids + [tokenizer.sep_token_id] + sub_inputs + [tokenizer.sep_token_id] + obj_inputs + [tokenizer.sep_token_id] # total max_len: 128+10+1+10+1=150
            attention_mask = [1] + context_mask + [1] + sub_mask + [1] + obj_mask + [1]
            token_type_ids = (MAX_CONTEXT_LEN + 1) * [0] + (MAX_NP_LEN * 2 + 3) * [1] # (len(input_ids) - len(token_type_ids)) * [1]
            token_type_ids = token_type_ids + (MAX_LEN - len(token_type_ids)) * [0]
            test_features1.append({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'context_len':context_len + 1})
            # load input for cls model
            context_input_ids = tokenizer(text, add_special_tokens = False, return_tensors='pt')['input_ids'][0].tolist()
            sub_inputs = tokenizer(np1, add_special_tokens = False, return_tensors='pt')['input_ids'][0].tolist()
            obj_inputs = tokenizer(np2, add_special_tokens = False, return_tensors='pt')['input_ids'][0].tolist()
            context_len = len(context_input_ids)
            input_ids = [tokenizer.cls_token_id] + context_input_ids + [s_id] + sub_inputs + [o_id] + obj_inputs + [tokenizer.sep_token_id] # total max_len: 128+10+1+10+1=150
            token_type_ids = (context_len + 2) * [0] + (len(sub_inputs) + 1) * [1] + (len(obj_inputs) + 1) * [1] # (len(input_ids) - len(token_type_ids)) * [1]
            attention_mask = (context_len + len(sub_inputs) + len(obj_inputs) + 4) * [1]
            input_len = len(input_ids)
            if input_len > 150:
                continue
            # padding
            pad_id = tokenizer.pad_token_id
            input_ids = input_ids + [pad_id] * (150 - len(input_ids))
            token_type_ids = token_type_ids + [0] * (150 - len(token_type_ids))
            attention_mask = attention_mask + [1] * (150 - len(attention_mask))
            # collect inputs
            inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
            # assert len(set([len(input_ids), len(token_type_ids), len(attention_masks), len(label_ids)])) == 1, 'incompatible length: %s, %s, %s, %s'%(len(input_ids), len(token_type_ids), len(attention_masks), len(label_ids))
            test_features2.append(inputs)
    test_dataloader1 = load_data1(test_features1, training=False, batch_size=16, gold=False)
    test_dataloader2 = load_data2(test_features2, training=False, batch_size=16, gold=False)

    config = BertConfig.from_pretrained("./bert-base-uncased",num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    intensive_module = IntensiveModule.from_pretrained("./bert-base-uncased", config=config)
    intensive_module.load_state_dict(torch.load('IntensiveModule.bin'))
    intensive_module.cuda()
    intensive_module.eval()

    test_pred_start = []
    test_pred_end = []
    test_confidence = []

    for batch in tqdm(list(test_dataloader1)):
        batch = [f.cuda() for f in batch]
        inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], 'context_lens': batch[-1]}
        with autocast():
            start_scores, end_scores = intensive_module(**inputs)
        start_scores = torch.softmax(start_scores,dim=2)[:,:,1].view(start_scores.shape[0], start_scores.shape[1])
        end_scores = torch.softmax(end_scores,dim=2)[:,:,1].view(end_scores.shape[0], end_scores.shape[1])

        batch_pred_start = torch.argmax(start_scores, dim=1) # bs*1
        batch_pred_end = torch.argmax(end_scores, dim=1) # bs*1
        batch_pred_start = batch_pred_start.flatten().detach().cpu().numpy().tolist()
        batch_pred_end = batch_pred_end.flatten().detach().cpu().numpy().tolist()
        start_conf = [start_scores[i, batch_pred_start[i]].item() for i in range(start_scores.shape[0])]
        end_conf = [end_scores[i, batch_pred_start[i]].item() for i in range(end_scores.shape[0])]
        batch_conf = [(sc + ec) / 2 for sc, ec in zip(start_conf, end_conf)]
        test_pred_start.extend(batch_pred_start)
        test_pred_end.extend(batch_pred_end)
        test_confidence.extend(batch_conf)

    config = BertConfig.from_pretrained("./bert-base-uncased",num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    extensive_module = ExtensiveModule.from_pretrained("./bert-base-uncased", config=config)
    extensive_module.load_state_dict(torch.load('ExtensiveModule.bin'))
    extensive_module.cuda()
    extensive_module.eval()
    confidence = []
    for batch in tqdm(list(test_dataloader2)):
        batch = [f.cuda() for f in batch]
        inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2]}
        with autocast():
            logits = extensive_module(**inputs)
        batch_conf = torch.softmax(logits, dim=1)[:,1]
        batch_conf = batch_conf.flatten().detach().cpu().numpy().tolist()
        confidence.extend(batch_conf)

    test_spans = []
    inverse_error = 0
    for s, e in zip(test_pred_start, test_pred_end):
        if e < s:
            e = s
            inverse_error+=1
        test_spans.append([s, e + 1])
    # inverse_error/len(test_spans)

    all_input_ids = [f['input_ids'] for f in test_features1]
    assert len(all_input_ids) == len(test_spans), 'unmatched input and output'
    result = {}
    for i in tqdm(range(len(test_spans))):
        doc_id = doc_ids[i]
        sentence, np1, np2 = input_text[i].split('[SEP]')
        if doc_id not in result.keys():
            result[doc_id] = []
        relation = get_phrase(all_input_ids[i], test_spans[i][0], test_spans[i][1])
        if (relation==None) or (len(relation.split(' '))>10):
            continue
        result[doc_id].append({"subject": np1, "relation": relation, "object": np2, "sentence": sentence, "score": confidence[i]})
    # rank results
    extraction_file = open('./data/test_output/extraction.txt', 'a',encoding='utf-8')
    extraction_file.truncate(0)
    sorted_keys = sorted(result.keys())
    K = 1
    for key in sorted_keys:
        if len(result[key]) > 0:
            result[key] = sorted(result[key],key=lambda x:x['score'],reverse=True)
            result[key] = result[key][:K]
            for top_triple in result[key]:
                extraction_file.write('\t'.join([top_triple['sentence'], str(top_triple['score']), top_triple['relation'], top_triple['subject'], top_triple['object']]))
                extraction_file.write('\n')
    extraction_file.close()

    output_path = './data/test_output'
    gold_path = './data/benchmarks/%s.oie'%TASK_NAME
    auc, precision, recall, f1 = [None for _ in range(4)]
    matching_func = Matcher.lexicalMatch
    error_fn = os.path.join(output_path, 'error_idxs.txt')

    evaluator = Benchmark(gold_path)
    reader = GeneralReader()
    reader.read(os.path.join('./data/test_output/extraction.txt'))
    (precision, recall, f1), auc = evaluator.compare(
        predicted=reader.oie,
        matchingFunc=matching_func,
        output_fn=os.path.join(output_path, 'pr_curve.txt'),
        error_file=error_fn)
    metric_names=["F1", "PREC", "REC", "AUC"]
    test_results=[f1, precision, recall, auc]
