import os
import warnings
warnings.filterwarnings("ignore")
import json
import pickle
from avro.datafile import DataFileReader
from avro.io import DatumReader
from itertools import combinations
from tqdm import tqdm
from random import shuffle
import random
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer
from utils import *

SPECIAL_TOKENS = True

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
def read_sents_from_avro(DATA_DIR):
    reader = DataFileReader(open(os.path.join(DATA_DIR, 'OPIEC-Clean-example.avro'), 'rb'), DatumReader())
    sent_dict = {}
    for triple in reader:
        # get sentence id
        if 'Wiki_https://en.wikipedia.org/wiki?curid=' in triple['article_id']:
            article_id = triple['article_id'].split('curid=')[-1]
        else:
            article_id = triple['article_id']
        sent_id = '@'.join([article_id, triple['sentence_number']])
        if sent_id not in sent_dict.keys():
            # add new sentence
            sent_dict[sent_id] = {}
            sent_dict[sent_id]['tokens'] = [token['word'] for token in triple['sentence_linked']['tokens']]
            sent_dict[sent_id]['pos_tags'] = [token['pos'] for token in triple['sentence_linked']['tokens']]
            sent_dict[sent_id]['triple_bounds'] = []
            sent_dict[sent_id]['related_np_pairs'] = []
        # add new triple
        sub_bounds = [triple['subject'][0]['index'] - 1, triple['subject'][-1]['index']]
        pred_bounds = [triple['relation'][0]['index'] - 1, triple['relation'][-1]['index']]
        obj_bounds = [triple['object'][0]['index'] - 1, triple['object'][-1]['index']]
        sent_dict[sent_id]['triple_bounds'].append([sub_bounds, pred_bounds, obj_bounds])
        # add np pair for matching
        sent_dict[sent_id]['related_np_pairs'].append([sub_bounds, obj_bounds])
    with open(os.path.join(DATA_DIR, 'sentences.json'), 'a') as f:
        f.truncate(0)
        json.dump(sent_dict, f)
    return sent_dict
def make_samples(sents, DATA_DIR, inverse_neg=False, neg_to_pos=False):
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    cls_inputs = []
    # get related entity pairs and unrelated entity pairs
    for sent in tqdm(sents):
        # get noun phrases
        sent_text = ' '.join(sent['tokens'])
        sent_spacy = nlp(sent_text)
        pos_tags = [token.pos_ for token in sent_spacy]
        np_bounds = [[chunk.start, chunk.end] for chunk in sent_spacy.noun_chunks]
        # np_bounds = [[ent.start, ent.end] for ent in sent_spacy.ents]
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
        related_np_pairs = [span_pair for span_pair in sent['related_np_pairs']]
        # split related entity pairs and unrelated entity pairs, respectively
        if inverse_neg:
            unrelated_np_pairs = [[obj_bounds, sub_bounds] for [sub_bounds, obj_bounds] in sent['related_np_pairs']]
        else:
            unrelated_np_pairs = []
        for span_pair in combinations(np_bounds, 2):
            span_pair = sort_span_pair(span_pair)
            if span_pair not in related_np_pairs:
                unrelated_np_pairs.append(span_pair)
        shuffle(unrelated_np_pairs)
        if neg_to_pos:
            unrelated_np_pairs = unrelated_np_pairs[:int(len(related_np_pairs)*neg_to_pos)]
        for np1_bounds, np2_bounds in related_np_pairs:
            np1 = sent['tokens'][np1_bounds[0]:np1_bounds[1]]
            np2 = sent['tokens'][np2_bounds[0]:np2_bounds[1]]
            cls_inputs.append([sent_text, ' '.join(np1), ' '.join(np2), 1])
        for np1_bounds, np2_bounds in unrelated_np_pairs:
            np1 = sent['tokens'][np1_bounds[0]:np1_bounds[1]]
            np2 = sent['tokens'][np2_bounds[0]:np2_bounds[1]]
            cls_inputs.append([sent_text, ' '.join(np1), ' '.join(np2), 0])
    shuffle(cls_inputs)
    # with open(os.path.join(DATA_DIR, 'cls_inputs.pkl'), 'wb') as f:
    #     pickle.dump(cls_inputs, f)
    return cls_inputs
def convert_examples_to_features(processed_data, max_len=150):
    features = []
    got = len(processed_data)
    kept = 0
    for sent_text, sub, obj, label in tqdm(processed_data):
        # get token ids aligned with labels
        context_input_ids = tokenizer(sent_text, add_special_tokens = False, return_tensors='pt')['input_ids'][0].tolist()
        sub_inputs = tokenizer(sub, add_special_tokens = False, return_tensors='pt')['input_ids'][0].tolist()
        obj_inputs = tokenizer(obj, add_special_tokens = False, return_tensors='pt')['input_ids'][0].tolist()
        context_len = len(context_input_ids)
        # append question to context
        if SPECIAL_TOKENS:
            input_ids = [tokenizer.cls_token_id] + context_input_ids + [s_id] + sub_inputs + [o_id] + obj_inputs + [tokenizer.sep_token_id] # total max_len: 128+10+1+10+1=150
        else:
            input_ids = [tokenizer.cls_token_id] + context_input_ids + [tokenizer.sep_token_id] + sub_inputs + [tokenizer.sep_token_id] + obj_inputs + [tokenizer.sep_token_id] # total max_len: 128+10+1+10+1=150
        token_type_ids = (context_len + 2) * [0] + (len(sub_inputs) + 1) * [1] + (len(obj_inputs) + 1) * [1] # (len(input_ids) - len(token_type_ids)) * [1]
        attention_mask = (context_len + len(sub_inputs) + len(obj_inputs) + 4) * [1]
        input_len = len(input_ids)
        if input_len > max_len:
            continue
        # padding
        pad_id = tokenizer.pad_token_id
        input_ids = input_ids + [pad_id] * (max_len - len(input_ids))
        token_type_ids = token_type_ids + [0] * (max_len - len(token_type_ids))
        attention_mask = attention_mask + [1] * (max_len - len(attention_mask))
        # collect inputs
        inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'label':label}
        # assert len(set([len(input_ids), len(token_type_ids), len(attention_masks), len(label_ids)])) == 1, 'incompatible length: %s, %s, %s, %s'%(len(input_ids), len(token_type_ids), len(attention_masks), len(label_ids))
        features.append(inputs)
        kept+=1
    print('got:', got, 'kept:', kept)
    return features

if __name__ == "__main__":
    DATA_DIR = './data'
    seed_everything()
    # sent_dict = read_sents_from_avro(DATA_DIR)
    with open(os.path.join(DATA_DIR, 'sentences.json'), 'r') as f:
        sent_dict = json.load(f)
    sents = list(sent_dict.values())

    train_sents = sents[:int(len(sents) * 0.8)]
    dev_sents = sents[int(len(sents) * 0.8):]
    train_inputs = make_samples(train_sents, DATA_DIR, neg_to_pos=10)
    dev_inputs = make_samples(dev_sents, DATA_DIR, neg_to_pos=None)

    num_samples = int(2e4)
    train_size = int(num_samples * 0.8)
    dev_size = num_samples - train_size
    train_inputs = train_inputs[:train_size]
    dev_inputs = dev_inputs[:dev_size]
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', do_lower_case = True, add_special_tokens = False, local_files_only = True)
    train_features = convert_examples_to_features(train_inputs)
    dev_featuers = convert_examples_to_features(dev_inputs)
    with open('./data/cached_features2.pkl', 'wb') as f:
        pickle.dump([train_features, dev_featuers], f)