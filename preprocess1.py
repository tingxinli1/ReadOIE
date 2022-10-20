import re
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader
from transformers import AutoTokenizer

SPECIAL_TOKENS = True
MAX_NP_LEN = 5
MAX_CONTEXT_LEN = 80
MAX_LEN = MAX_NP_LEN * 2 + MAX_CONTEXT_LEN + 4
QUERY_LEN = MAX_NP_LEN * 2 + 2

if SPECIAL_TOKENS:
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, add_special_tokens = False, local_files_only = True,)
    # additional_special_tokens = ['[#S]', '[#O]', '[#Context]'])
    s_id = tokenizer.vocab['[#S]']
    o_id = tokenizer.vocab['[#O]']
    context_id = tokenizer.vocab['[#Context]']
else:
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, add_special_tokens = False, local_files_only = True)

class Triple:
    def __init__(self, tagged_sent, max_context_len=MAX_CONTEXT_LEN, max_np_len=MAX_NP_LEN, DEBUG=False):
        self.token_list = [] # ['-', '[CLS]', tokenizer.cls_token_id, 0, 1, '-']
        for raw_token, tag in tagged_sent:
            encoded_input = tokenizer(raw_token, add_special_tokens = False, return_tensors='pt')
            input_ids = encoded_input['input_ids'][0].tolist()
            token_type_ids = encoded_input['token_type_ids'][0].tolist()
            attention_masks = encoded_input['attention_mask'][0].tolist()
            tokens = [tokenizer.decode(id) for id in encoded_input['input_ids'][0]]
            rep = 0
            for input_id, token_type_id, attention_mask, token in zip(input_ids, token_type_ids, attention_masks, tokens):
                if rep > 0:
                    self.token_list.append(['[SUF]', token, input_id, token_type_id, attention_mask, tag])
                else:
                    self.token_list.append([raw_token, token, input_id, token_type_id, attention_mask, tag])
                    rep+=1
        # self.token_list.append(['-', '[SEP]', tokenizer.sep_token_id, 0, 1, '-'])
        self.token_list = pd.DataFrame(self.token_list, columns = ['raw_token', 'token', 'input_id', 'token_type_id', 'attention_mask', 'tag'])
        self.raw_tokens = self.token_list.raw_token.to_list()
        tags = self.token_list.tag.to_list()
        self.triple_spans = {}
        i = 0
        while i < len(tags):
            span_left = 0
            span_right = 0
            if tags[i] != '-':
                span_left = i
                span_right = i + 1
                while span_right < len(tags):
                    if tags[span_left] == tags[span_right]:
                        span_right+=1
                    else:
                        break
                self.triple_spans[tags[i]] = [span_left, span_right]
                i = span_right
            else:
                i+=1
        # make inputs
        context_input_ids = self.token_list.input_id.to_list()
        token_type_ids = self.token_list.token_type_id.to_list()
        attention_mask = self.token_list.attention_mask.to_list()
        start_positions = [0] * (max_context_len + 1)
        end_positions = [0] * (max_context_len + 1)
        # question inputs
        sub_left, sub_right = self.triple_spans['S']
        sub_inputs = context_input_ids[sub_left:sub_right]
        obj_left, obj_right = self.triple_spans['O']
        obj_inputs = context_input_ids[obj_left:obj_right]
        # answer label
        try:
            pred_left, pred_right = self.triple_spans['P']
            start_positions[pred_left+1] = 1
            end_positions[pred_right] = 1
        except:# KeyError case of no answer
            pass
        context_len = len(context_input_ids)
        assert context_len < max_context_len, "Input length overflowed, got %d, expected %d.\n Trigger source: %s"%(context_len, max_context_len, self.show_content())
        pad_id = tokenizer.pad_token_id
        sub_inputs = sub_inputs[:max_np_len]
        sub_mask = [1] * (len(sub_inputs)) + [0] * (max_np_len - len(sub_inputs))
        sub_inputs = sub_inputs + [pad_id] * (max_np_len - len(sub_inputs))
        obj_inputs = obj_inputs[:max_np_len]
        obj_mask = [1] * (len(obj_inputs)) + [0] * (max_np_len - len(obj_inputs))
        obj_inputs = obj_inputs + [pad_id] * (max_np_len - len(obj_inputs))
        context_mask = [1] * len(context_input_ids) + [0] * (max_context_len - len(context_input_ids))
        context_input_ids = context_input_ids + [pad_id] * (max_context_len - len(context_input_ids))
        # append question to context
        if SPECIAL_TOKENS:
            input_ids = [tokenizer.cls_token_id] + context_input_ids + [s_id] + sub_inputs + [o_id] + obj_inputs + [tokenizer.sep_token_id] # total max_len: 128+10+1+10+1=150
        else:
            input_ids = [tokenizer.cls_token_id] + context_input_ids + [tokenizer.sep_token_id] + sub_inputs + [tokenizer.sep_token_id] + obj_inputs + [tokenizer.sep_token_id] # total max_len: 128+10+1+10+1=150
        attention_mask = [1] + context_mask + [1] + sub_mask + [1] + obj_mask + [1]
        token_type_ids = (context_len + 1) * [0] + (max_np_len * 2 + 3) * [1] # (len(input_ids) - len(token_type_ids)) * [1]
        token_type_ids = token_type_ids + (MAX_LEN - len(token_type_ids)) * [0]
        
        # collect inputs
        self.inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'start_positions': start_positions, 'end_positions': end_positions, 'context_len': context_len + 2}
        if not DEBUG:
            self.token_list = None

    def show_content(self):
        sent = ' '.join(self.raw_tokens[1:-1])
        sub_left, sub_right = self.triple_spans['S']
        sub_str = ' '.join(self.raw_tokens[sub_left:sub_right])
        try:
            pred_left, pred_right = self.triple_spans['P']
            pred_str = ' '.join(self.raw_tokens[pred_left:pred_right])
        except KeyError:
            pred_str = 'NO RELATION'
        obj_left, obj_right = self.triple_spans['O']
        obj_str = ' '.join(self.raw_tokens[obj_left:obj_right])
        content = '\t'.join([sent, sub_str, pred_str, obj_str])
        content = re.sub('\[SUF\]', '', content)
        content = re.sub('  ', ' ', content)
        return content

if __name__ == "__main__":
    print("Reading examples from database...")
    # read data
    AVRO_FILE = "./data/OPIEC-Clean-example.avro"
    reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
    tag_file = open('./data/tag_file.txt', 'a', encoding='utf-8')
    tag_file.truncate(0)
    num_triples = 0
    num_valid_triples = 0
    max_len = 0
    for triple in reader:
        # get sentence
        sent_tokens = [token['word'] for token in triple['sentence_linked']['tokens']]
        # compute max_len
        if len(sent_tokens) > max_len:
            max_len = len(sent_tokens)
        # get pos tags
        sent_pos = [token['pos'] for token in triple['sentence_linked']['tokens']]
        # tag triple
        sub_bounds = [triple['subject'][0]['index'] - 1, triple['subject'][-1]['index']]
        pred_bounds = [triple['relation'][0]['index'] - 1, triple['relation'][-1]['index']]
        obj_bounds = [triple['object'][0]['index'] - 1, triple['object'][-1]['index']]
        SPO_tags = np.array(['-'] * len(sent_tokens))
        SPO_tags[sub_bounds[0]:sub_bounds[1]] = 'S'
        SPO_tags[pred_bounds[0]:pred_bounds[1]] = 'P'
        SPO_tags[obj_bounds[0]:obj_bounds[1]] = 'O'
        pred_tags = ' '.join(sent_pos[pred_bounds[0]:pred_bounds[1]])
        num_triples+=1
        # delete triples that are incomplete or with non-verbal relations
        if len(set(SPO_tags)) < 4: #  or 'VB' not in pred_tags
            continue
        num_valid_triples+=1
        # write tag file
        for token, tag in zip(sent_tokens, SPO_tags):
            tag_file.write('\t'.join([token, tag]))
            tag_file.write('\n')
        tag_file.write('\n')
    tag_file.close()
    reader.close()
    # print("    max_len:", max_len)
    print("    kept:", num_valid_triples / num_triples)
    
    print("Converting examples into input features...")
    # tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, add_special_tokens = False, local_files_only = True)
    # MAX_LEN = 128
    tag_file = open('./data/tag_file.txt', 'r', encoding='utf-8')
    sent = []
    sents = []
    for line in tag_file.readlines():
        if line != '\n':
            sent.append(line.strip().split('\t')) # raw_token, tag
        else:
            sents.append(sent.copy())
            sent.clear()
    tag_file.close()
    features = []
    num_overflowed = 0
    # sent: [raw_token, tag]
    for sent in tqdm(sents):
        try:
            features.append(Triple(sent).inputs)
        except AssertionError:
            num_overflowed+=1
    print("    got %d sentences, dropped %d because of overflow"%(len(sents), num_overflowed))
    with open('./data/cached_features1.pkl', 'wb') as f:
        pickle.dump(features, f)
    