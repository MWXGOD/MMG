import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
import logging
from math import log
import config_yuanshi_start_end_mul_out_new_decoding as config
import data_loader_start_end_mul_out_new_decoding as data_loader
import utils
from model_att_start_end_mul_out_new_decoding import Model
import sys
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']

        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        # 学习率预热
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader, args):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
        
        bnw_label_result = []
        epw_label_result = []
        
        pred_result_BNW = []
        pred_result_EPW = []

        
            
        
        
        for i, data_batch in enumerate(tqdm(data_loader)):
            data_batch = [data.cuda() if (i != 6 and i !=15 and i!=16 ) else data for i, data in enumerate(data_batch)]


            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, _, grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels, grid_end_2_labels, grid_bnw_labels, grid_epw_labels, _, _, adjacency_matrix= data_batch
            outputs, outputs2, outputs3, outputs_BNW, outputs_EPW = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, adjacency_matrix) #, sigma_1, sigma_2, matrix_embeddings

            
            grid_mask2d = grid_mask2d.clone()
            loss1 = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
            loss_BNW = self.criterion(outputs_BNW[grid_mask2d], grid_bnw_labels[grid_mask2d])
            loss_EPW = self.criterion(outputs_EPW[grid_mask2d], grid_epw_labels[grid_mask2d])
            
            loss6 = self.criterion(outputs2[grid_mask2d], grid_sta_2_labels[grid_mask2d])
            loss7 = self.criterion(outputs3[grid_mask2d], grid_end_2_labels[grid_mask2d])
            
            loss = loss_BNW+ loss_EPW + args.loss2_weight*loss6+args.loss3_weight*loss7

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            grid_bnw_labels = grid_bnw_labels[grid_mask2d].contiguous().view(-1)
            grid_epw_labels = grid_epw_labels[grid_mask2d].contiguous().view(-1)
            
            outputs_BNW = torch.argmax(outputs_BNW, -1)
            outputs_BNW = outputs_BNW[grid_mask2d].contiguous().view(-1)
            
            outputs_EPW = torch.argmax(outputs_EPW, -1)
            outputs_EPW = outputs_EPW[grid_mask2d].contiguous().view(-1)
            

            # label_result.append(grid_labels.cpu())
            bnw_label_result.append(grid_bnw_labels.cpu())
            epw_label_result.append(grid_epw_labels.cpu())
            pred_result_BNW.append(outputs_BNW.cpu())
            pred_result_EPW.append(outputs_EPW.cpu())

            self.scheduler.step()
        bnw_label_result = torch.cat(bnw_label_result)
        epw_label_result = torch.cat(epw_label_result)
        
        pred_result_BNW = torch.cat(pred_result_BNW)
        pred_result_EPW = torch.cat(pred_result_EPW)

        p_BNW, r_BNW, f1_BNW, _ = precision_recall_fscore_support(bnw_label_result.numpy(),
                                                      pred_result_BNW.numpy(),
                                                      average="macro")
        
        p_EPW, r_EPW, f1_EPW, _ = precision_recall_fscore_support(epw_label_result.numpy(),
                                                      pred_result_EPW.numpy(),
                                                      average="macro")
        

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label_BNW", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1_BNW, p_BNW, r_BNW]])
        table.add_row(["Label_EPW", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1_EPW, p_EPW, r_EPW]])
        logger.info("\n{}".format(table))
        return f1_BNW, f1_EPW

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result_BNW = []
        pred_result_EPW = []
        bnw_label_result = []
        epw_label_result = []

        total_ent_r_BNW = 0
        total_ent_p_BNW = 0
        total_ent_c_BNW = 0

        total_ent_r_EPW = 0
        total_ent_p_EPW = 0
        total_ent_c_EPW = 0
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(data_loader)):
                
                entity_text = data_batch[6]
                data_batch = [data.cuda() if (i != 6 and i !=15 and i!=16 ) else data for i, data in enumerate(data_batch)]
                
                
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, _, grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels, grid_end_2_labels, grid_bnw_labels, grid_epw_labels, _, _, adjacency_matrix= data_batch

                outputs, _, _, outputs_BNW, outputs_EPW = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, adjacency_matrix)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()
                
                outputs_BNW = torch.argmax(outputs_BNW, -1)
                outputs_EPW = torch.argmax(outputs_EPW, -1)

                ent_c_BNW, ent_p_BNW, ent_r_BNW, _ = utils.BNW_decode(outputs_BNW.cpu().numpy(), entity_text, length.cpu().numpy())
                ent_c_EPW, ent_p_EPW, ent_r_EPW, _ = utils.EPW_decode(outputs_EPW.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r_BNW += ent_r_BNW
                total_ent_p_BNW += ent_p_BNW
                total_ent_c_BNW += ent_c_BNW
                
                total_ent_r_EPW += ent_r_EPW
                total_ent_p_EPW += ent_p_EPW
                total_ent_c_EPW += ent_c_EPW
                
                grid_bnw_labels = grid_bnw_labels[grid_mask2d].contiguous().view(-1)
                grid_epw_labels = grid_epw_labels[grid_mask2d].contiguous().view(-1)
                
                outputs_BNW = outputs_BNW[grid_mask2d].contiguous().view(-1)
                outputs_EPW = outputs_EPW[grid_mask2d].contiguous().view(-1)
                
                bnw_label_result.append(grid_bnw_labels.cpu())
                epw_label_result.append(grid_epw_labels.cpu())
                
                pred_result_BNW.append(outputs_BNW.cpu())
                pred_result_EPW.append(outputs_EPW.cpu())
                
        bnw_label_result = torch.cat(bnw_label_result)
        epw_label_result = torch.cat(epw_label_result)
        
        pred_result_BNW = torch.cat(pred_result_BNW)
        pred_result_EPW = torch.cat(pred_result_EPW)
        
        p_BNW, r_BNW, f1_BNW, _ = precision_recall_fscore_support(bnw_label_result.numpy(),
                                                      pred_result_BNW.numpy(),
                                                      average="macro")

        p_EPW, r_EPW, f1_EPW, _ = precision_recall_fscore_support(epw_label_result.numpy(),
                                                      pred_result_EPW.numpy(),
                                                      average="macro")
        
        e_f1_BNW, e_p_BNW, e_r_BNW = utils.cal_f1(total_ent_c_BNW, total_ent_p_BNW, total_ent_r_BNW)
        e_f1_EPW, e_p_EPW, e_r_EPW = utils.cal_f1(total_ent_c_EPW, total_ent_p_EPW, total_ent_r_EPW)
        
        title = "EVAL" if not is_test else "TEST"
        logger.info('{}_BNW Label F1 {}'.format(title, f1_score(bnw_label_result.numpy(),
                                                            pred_result_BNW.numpy(),
                                                            average=None)))
        logger.info('{}_EPW Label F1 {}'.format(title, f1_score(epw_label_result.numpy(),
                                                            pred_result_EPW.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label_BNW"] + ["{:3.4f}".format(x) for x in [f1_BNW, p_BNW, r_BNW]])
        table.add_row(["Label_EPW"] + ["{:3.4f}".format(x) for x in [f1_EPW, p_EPW, r_EPW]])
        table.add_row(["Entity_BNW"] + ["{:3.4f}".format(x) for x in [e_f1_BNW, e_p_BNW, e_r_BNW]])
        table.add_row(["Entity_EPW"] + ["{:3.4f}".format(x) for x in [e_f1_EPW, e_p_EPW, e_r_EPW]])

        logger.info("\n{}".format(table))
        return e_f1_BNW, e_f1_EPW

    def predict(self, epoch, data_loader, data):
        self.model.eval()

        pred_result_BNW = []
        pred_result_EPW = []
        bnw_label_result = []
        epw_label_result = []

        result_BNW = []
        result_EPW = []

        total_ent_r_BNW = 0
        total_ent_p_BNW = 0
        total_ent_c_BNW = 0

        total_ent_r_EPW = 0
        total_ent_p_EPW = 0
        total_ent_c_EPW = 0

        i = 0
        with torch.no_grad():
            for data_batch in tqdm(data_loader):
                sentence_batch = data[i:i+config.batch_size]
                
                entity_text = data_batch[6]
                data_batch = [data.cuda() if (i != 6 and i !=15 and i!=16 ) else data for i, data in enumerate(data_batch)]
                
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, _, grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels, grid_end_2_labels, grid_bnw_labels, grid_epw_labels, _, _, adjacency_matrix= data_batch      

                
                outputs, _, _, outputs_BNW, outputs_EPW = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, adjacency_matrix)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()
                
                outputs_BNW = torch.argmax(outputs_BNW, -1)
                outputs_EPW = torch.argmax(outputs_EPW, -1)
                ent_c_BNW, ent_p_BNW, ent_r_BNW, decode_entities_BNW = utils.BNW_decode(outputs_BNW.cpu().numpy(),
                                                                                    entity_text, length.cpu().numpy())
                ent_c_EPW, ent_p_EPW, ent_r_EPW, decode_entities_EPW = utils.EPW_decode(outputs_EPW.cpu().numpy(), 
                                                                                    entity_text, length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities_BNW, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        if len(ent[0])==1:
                            instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                       "type": config.vocab.id_to_label(ent[1])})
                        else:
                            instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                       "type": config.vocab.id_to_label(ent[1])})
                    result_BNW.append(instance)
                    
                for ent_list, sentence in zip(decode_entities_EPW, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        if len(ent[0])==1:
                            instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                       "type": config.vocab.id_to_label(ent[1])})
                        else:
                            instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                       "type": config.vocab.id_to_label(ent[1])})
                    result_EPW.append(instance)

                total_ent_r_BNW += ent_r_BNW
                total_ent_p_BNW += ent_p_BNW
                total_ent_c_BNW += ent_c_BNW
                
                total_ent_r_EPW += ent_r_EPW
                total_ent_p_EPW += ent_p_EPW
                total_ent_c_EPW += ent_c_EPW

                grid_bnw_labels = grid_bnw_labels[grid_mask2d].contiguous().view(-1)
                grid_epw_labels = grid_epw_labels[grid_mask2d].contiguous().view(-1)
                
                outputs_BNW = outputs_BNW[grid_mask2d].contiguous().view(-1)
                outputs_EPW = outputs_EPW[grid_mask2d].contiguous().view(-1)

                bnw_label_result.append(grid_bnw_labels.cpu())
                epw_label_result.append(grid_epw_labels.cpu())
                pred_result_BNW.append(outputs_BNW.cpu())
                pred_result_EPW.append(outputs_EPW.cpu())

                i += config.batch_size

        bnw_label_result = torch.cat(bnw_label_result)
        epw_label_result = torch.cat(epw_label_result)
        pred_result_BNW = torch.cat(pred_result_BNW)
        pred_result_EPW = torch.cat(pred_result_EPW)

        
        p_BNW, r_BNW, f1_BNW, _ = precision_recall_fscore_support(bnw_label_result.numpy(),
                                                      pred_result_BNW.numpy(),
                                                      average="macro")

        p_EPW, r_EPW, f1_EPW, _ = precision_recall_fscore_support(epw_label_result.numpy(),
                                                      pred_result_EPW.numpy(),
                                                      average="macro")

        e_f1_BNW, e_p_BNW, e_r_BNW = utils.cal_f1(total_ent_c_BNW, total_ent_p_BNW, total_ent_r_BNW)
        e_f1_EPW, e_p_EPW, e_r_EPW = utils.cal_f1(total_ent_c_EPW, total_ent_p_EPW, total_ent_r_EPW)
        
        title = "TEST"
        logger.info('{}_BNW Label F1 {}'.format(title, f1_score(bnw_label_result.numpy(),
                                                            pred_result_BNW.numpy(),
                                                            average=None)))
        logger.info('{}_EPW Label F1 {}'.format(title, f1_score(epw_label_result.numpy(),
                                                            pred_result_EPW.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label_BNW"] + ["{:3.4f}".format(x) for x in [f1_BNW, p_BNW, r_BNW]])
        table.add_row(["Label_EPW"] + ["{:3.4f}".format(x) for x in [f1_EPW, p_EPW, r_EPW]])
        table.add_row(["Entity_BNW"] + ["{:3.4f}".format(x) for x in [e_f1_BNW, e_p_BNW, e_r_BNW]])
        table.add_row(["Entity_EPW"] + ["{:3.4f}".format(x) for x in [e_f1_EPW, e_p_EPW, e_r_EPW]])

        logger.info("\n{}".format(table))

        with open(config.predict_path+'_BNW', "w", encoding="utf-8") as f:
            json.dump(result_BNW, f, ensure_ascii=False)
        with open(config.predict_path+'_EPW', "w", encoding="utf-8") as f:
            json.dump(result_EPW, f, ensure_ascii=False)

        return e_f1_BNW, e_f1_EPW

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/BC5CDR.json')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--device', type=int, default=0)
    
    parser.add_argument('--test_dep_reg_dis', type=int, default=0)
    parser.add_argument('--test_conv', type=int, default=0)
    
    parser.add_argument('--is_pre_Fine_tuning', type=str, default='False')
    parser.add_argument('--pre_Fine_tuning_Model', type=str, default='')
    parser.add_argument('--shot', type=str, default='')
    parser.add_argument('--DA_time', type=int, default=20)
    parser.add_argument('--data_set', type=str, default='')
    
    parser.add_argument('--loss2_weight', type=float, default=0.1)
    parser.add_argument('--loss3_weight', type=float, default=0.1)
    parser.add_argument('--loss_pos', type=float, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)# , default=0.0001
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    
    config = config.Config(args)
    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    print('---------------------------------------------')
    print(f'{config.dataset}/{config.shot}')
    print('---------------------------------------------')
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size, # config.batch_size, 
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )
    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    
    model = Model(config)
#      
    if config.is_pre_Fine_tuning == 'False':
        if config.pre_Fine_tuning_Model == '':
            model.load_state_dict(torch.load('_BNW_final_ontonotes5_model.pt'), strict=False) # 默认加载onto作为预微调模型
        else:
            model.load_state_dict(torch.load(config.pre_Fine_tuning_Model), strict=False)
    else:


    model = model.cuda()
    
    
    for i, data_batch in enumerate(tqdm(data_loader)):
        data_batch = [data.cuda() if (i != 6 and i !=15 and i!=16 ) else data for i, data in enumerate(data_batch)]


        bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, _, grid_sta_labels, grid_end_labels, grid_sta_1_labels, grid_end_1_labels, grid_sta_2_labels, grid_end_2_labels, grid_bnw_labels, grid_epw_labels, _, _, adjacency_matrix= data_batch
        word_reps = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, adjacency_matrix) #, sigma_1, sigma_2, matrix_embeddings
    
        