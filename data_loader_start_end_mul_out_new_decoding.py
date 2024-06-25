import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9



class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC1 = '<suc1>'
    SUC3 = '<suc3>'
    SUC5 = '<suc5>'
    SUC7 = '<suc7>'

    def __init__(self):
        # self.label2id = {self.PAD: 0, self.SUC1: 1, self.SUC3: 2, self.SUC5: 3, self.SUC7: 4 }
        # self.id2label = {0: self.PAD, 1: self.SUC1, 2: self.SUC3, 3: self.SUC5, 4: self.SUC7 }
        self.label2id = {self.PAD: 0, self.SUC1: 1, self.SUC3: 2 }
        self.id2label = {0: self.PAD, 1: self.SUC1, 2: self.SUC3 }

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            lable_len = len(self.label2id)
            self.label2id[label] = lable_len
            self.id2label[self.label2id[label]] = label
            
        if label != self.id2label[self.label2id[label]]:
            print(label != self.id2label[self.label2id[label]])
        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text, grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels, grid_end_2_labels, grid_bnw_labels, grid_epw_labels, entity_bnw_text, entity_epw_text, adjacency_matrix = map(list, zip(*data)) 

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)
    
    def fill(data, new_data):
        for j, x in enumerate(data):
            new_x = x[:new_data.shape[1], :new_data.shape[2]]
            new_data[j, :new_x.shape[0], :new_x.shape[1]] = new_x
        return new_data

    
    def fill_2d(data, new_data):
        for j, x in enumerate(data):
            new_x = x[:new_data.shape[1]]
            new_data[j, :new_x.shape[0]] = new_x
        return new_data


    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    labels_sta_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_sta_labels = fill(grid_sta_labels, labels_sta_mat)
    labels_end_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_end_labels = fill(grid_end_labels, labels_end_mat)

    labels_sta_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_sta_1_labels = fill(grid_sta_labels, labels_sta_mat)
    labels_end_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_end_1_labels = fill(grid_end_labels, labels_end_mat)

    labels_sta_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_sta_2_labels = fill(grid_sta_labels, labels_sta_mat)
    labels_end_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_end_2_labels = fill(grid_end_labels, labels_end_mat)

    labels_bnw_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_bnw_labels = fill(grid_bnw_labels, labels_bnw_mat)
    labels_epw_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_epw_labels = fill(grid_epw_labels, labels_epw_mat)
    
    adjacency_mat = torch.zeros((batch_size, max_tok+1, max_tok+1), dtype=torch.long)
    grid_adjacency = fill(adjacency_matrix, adjacency_mat)
    pos_label_mat = torch.zeros((batch_size, max_tok), dtype=torch.long)
    

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text, grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels, grid_end_2_labels, grid_bnw_labels, grid_epw_labels, entity_bnw_text, entity_epw_text, grid_adjacency# , pos_label#, input_mask_1D, input_type_ids


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text,
                 grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels,
                 grid_end_2_labels, grid_bnw_labels, grid_epw_labels, entity_bnw_text, entity_epw_text, adjacency_matrix):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text
        self.grid_sta_labels = grid_sta_labels
        self.grid_end_labels = grid_end_labels
        self.grid_sta_1_labels = grid_sta_1_labels
        self.grid_end_1_labels = grid_end_1_labels
        self.grid_sta_2_labels = grid_sta_2_labels
        self.grid_end_2_labels = grid_end_2_labels
        self.grid_bnw_labels = grid_bnw_labels
        self.grid_epw_labels = grid_epw_labels
        self.entity_bnw_text = entity_bnw_text
        self.entity_epw_text = entity_epw_text
        self.adjacency_matrix = adjacency_matrix

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
            torch.LongTensor(self.grid_labels[item]), \
            torch.LongTensor(self.grid_mask2d[item]), \
            torch.LongTensor(self.pieces2word[item]), \
            torch.LongTensor(self.dist_inputs[item]), \
            self.sent_length[item], \
            self.entity_text[item], \
            torch.LongTensor(self.grid_sta_labels[item]), \
            torch.LongTensor(self.grid_end_labels[item]), \
            torch.LongTensor(self.grid_sta_1_labels[item]), \
            torch.LongTensor(self.grid_end_1_labels[item]), \
            torch.LongTensor(self.grid_sta_2_labels[item]), \
            torch.LongTensor(self.grid_end_2_labels[item]), \
            torch.LongTensor(self.grid_bnw_labels[item]), \
            torch.LongTensor(self.grid_epw_labels[item]), \
            self.entity_bnw_text[item], \
            self.entity_epw_text[item], \
            torch.LongTensor(self.adjacency_matrix[item]), \

    def __len__(self):
        return len(self.bert_inputs)



def is_incremental_by_one(lst):
    if len(lst) <= 1:
        return True
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] != 1:
            return False
    return True

def process_bert(data, tokenizer, vocab):
    bert_inputs = []
    
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    entity_bnw_text = []
    entity_epw_text = []
    pieces2word = []
    sent_length = []
    grid_sta_labels = []
    grid_end_labels = []
    grid_sta_1_labels = []
    grid_end_1_labels = []
    grid_sta_2_labels = []
    grid_end_2_labels = []
    grid_bnw_labels = []
    grid_epw_labels = []
    
    adjacency_matrix = []
    pos_label = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        _entity_bnw_text = []
        _entity_epw_text = []
        
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        

        length = len(instance['sentence'])
        if len(_bert_inputs) > 512:
            print('index:', index)
            print('sentence:', ' '.join(instance['sentence']))
            continue
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        _grid_sta_labels = np.zeros((length, length), dtype=np.int)
        _grid_end_labels = np.zeros((length, length), dtype=np.int)

        _grid_sta_1_labels = np.zeros((length, length), dtype=np.int)
        _grid_end_1_labels = np.zeros((length, length), dtype=np.int)

        _grid_sta_2_labels = np.zeros((length, length), dtype=np.int)
        _grid_end_2_labels = np.zeros((length, length), dtype=np.int)

        _grid_BNW_labels = np.zeros((length, length), dtype=np.int)
        _grid_EPW_labels = np.zeros((length, length), dtype=np.int)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            sta = index[0]
            end_ = index[-1]
            if end_>=length:
                break
            for i in range(sta, end_):
                if i + 1 >= length:
                    break
                _grid_sta_labels[i, i + 1] = 1
            if sta - 1 >= 0:
                _grid_sta_labels[sta - 1, sta] = 1
            if sta - 2 >= 0:
                _grid_sta_labels[sta - 2, sta - 1] = 1
            if sta - 2 >= 0:
                _grid_sta_2_labels[end_, sta - 2:end_ + 1] = vocab.label_to_id(entity["type"])
            if sta - 1 >= 0:
                _grid_sta_2_labels[end_, sta - 1:end_ + 1] = vocab.label_to_id(entity["type"])
            if sta == 0:
                _grid_sta_2_labels[end_, sta:end_ + 1] = vocab.label_to_id(entity["type"])

        for entity in instance["ner"]:
            index = entity["index"]
            sta = index[0]
            end_ = index[-1]
            if end_>=length:
                break
            for i in range(sta, end_):
                if i + 1 >= length:
                    break
                _grid_sta_labels[i, i + 1] = 1
            if end_ + 1 < length:
                _grid_sta_labels[end_, end_ + 1] = 1
            if end_ + 2 < length:
                _grid_sta_labels[end_ + 1, end_ + 2] = 1
            if end_ + 2 < length:
                _grid_end_2_labels[sta:end_ + 3, sta] = vocab.label_to_id(entity["type"])
                continue
            if end_ + 1 < length:
                _grid_end_2_labels[sta:end_ + 2, sta] = vocab.label_to_id(entity["type"])
                continue
            if end_ == length:
                _grid_end_2_labels[sta:end_, sta] = vocab.label_to_id(entity["type"])

        for entity in instance["ner"]:
            index = entity["index"]
            if index[-1]>=length:
                continue
            for i in range(len(index)):
                if i+1>=len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        for entity in instance["ner"]:
            index = entity["index"]
            sta, end = index[0], index[-1]
            if end>=length:
                break
            if sta == end:
                _grid_BNW_labels[index[0], index[0]] = vocab.label_to_id(entity["type"])  # +99
            else:
                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    _grid_BNW_labels[index[i], index[i + 1]] = 2
                _grid_BNW_labels[index[0], index[1]] = 1
                if is_incremental_by_one(index):
                    _grid_BNW_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])
                else:
                    _grid_BNW_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"] + '_discontinuous')

        for entity in instance["ner"]:
            index = entity["index"]
            sta, end = index[0], index[-1]
            if end>=length:
                break
            if sta == end:
                _grid_EPW_labels[index[0], index[0]] = vocab.label_to_id(entity["type"])  # +99
            else:
                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    _grid_EPW_labels[index[i], index[i + 1]] = 2
                _grid_EPW_labels[index[-2], index[-1]] = 1
                if is_incremental_by_one(index):
                    _grid_EPW_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])
                else:
                    _grid_EPW_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"] + '_discontinuous')

        for e in instance["ner"]:
            index = entity["index"]
            sta, end = index[0], index[-1]
            if end>=length:
                break
            _entity_bnw_text.append(utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"])))
        _entity_bnw_text = set(_entity_bnw_text)

        for e in instance["ner"]:
            index = entity["index"]
            sta, end = index[0], index[-1]
            if end>=length:
                break
            _entity_epw_text.append(utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"])))
        _entity_epw_text = set(_entity_epw_text)            
                    
        
        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])
        
        
        sent_length.append(length)
        bert_inputs.append(_bert_inputs)

        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)
        grid_sta_labels.append(_grid_sta_labels)
        grid_end_labels.append(_grid_sta_labels)

        grid_sta_1_labels.append(_grid_sta_1_labels)
        grid_end_1_labels.append(_grid_end_1_labels)
        grid_sta_2_labels.append(_grid_sta_2_labels)
        grid_end_2_labels.append(_grid_end_2_labels)

        grid_bnw_labels.append(_grid_BNW_labels)
        grid_epw_labels.append(_grid_EPW_labels)
        entity_bnw_text.append(_entity_bnw_text)
        entity_epw_text.append(_entity_epw_text)
        
        adjacency_matrix.append(np.array(instance['adjacency_matrix']))

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text, grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels, grid_end_2_labels, grid_bnw_labels, grid_epw_labels, entity_bnw_text, entity_epw_text, adjacency_matrix#, pos_label#, input_mask_1D, input_type_ids


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(config):
    with open(f'./data/{config.data_set}{config.shot}/train_dep_pos.json', 'r', encoding='utf-8') as f:  # _best_clear
        train_data = json.load(f)
    with open(f'./data/{config.data_set}{config.shot}/dev_dep_pos.json', 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(f'./data/{config.data_set}/test_dep_pos.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    
    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    print(vocab.label2id)
    print("len(vocab.label2id):", len(vocab.label2id))
    config.vocab = vocab
    

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))

    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)

