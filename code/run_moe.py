from __future__ import absolute_import, division, print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import random
import sys
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
import torch.nn.functional
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME
from transformers import RobertaTokenizer, RobertaConfig
from pytorch_pretrained_bert.optimization import BertAdam
from tensorboardX import SummaryWriter
from model import  RobertaMoEForSequenceClassification
import nltk
import logging
import pandas as pd

logger = logging.getLogger(__name__)

entity_linking_pattern = re.compile('\u2726.*?\u25C6-*[0-9]+,(-*[0-9]+)\u2726')
fact_pattern = re.compile('\u2726(.*?)\u25C6-*[0-9]+,-*[0-9]+\u2726')
unk_pattern = re.compile('\u2726([^\u2726]+)\u25C6-1,-1\u2726')
TSV_DELIM = "\t"
TBL_DELIM = ";"
months = ['january','february','march','april','may','june','july','august','september','october','november','december']
months_abbr = ['jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','dec',
                'jan.','feb.','mar.','apr.','may.','jun.','jul.','aug.','sept.','oct.','nov.','dec.']

def parse_fact(fact):
    chunks = re.split(fact_pattern, fact)
    output = ' '.join([x.strip() for x in chunks if len(x.strip()) > 0])
    return output


class InputExample(object):
    def __init__(self, table_name, text_a, text_b=None, label=None,priori=None):
        '''
        Args:
            guid:   unique id
            text_a: statement
            text_b: table_str
            label:  positive / negative
            priori: priori distribution over experts based on rules
        '''
        self.table_name = table_name
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.priori = priori


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, priori):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.priori = priori


class DataProcessor(object):
    def get_examples(self, data_dir, dataset=None):
        logger.info('Get examples from: {}.tsv'.format(dataset))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset))))

    def get_labels(self):
        return [0, 1], len([0, 1])

    def _read_tsv(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for idx, line in enumerate(f):
                entry = line.strip().split('\t')
                table_name = entry[0]
                table_str = parse_fact(entry[3])
                statement = entry[4]
                label = int(entry[5])
                priori = get_priori_distribution(statement,table_name,T=1)
                lines.append([table_name, statement, table_str, label, priori])
                if args.debug_mode and idx==127:
                    break
            return lines


    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            table_name = line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            priori = line[4]
            examples.append((InputExample(table_name=table_name, text_a=text_a, text_b=text_b, label=label, priori=priori)))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(tqdm(examples, desc="convert to features")):

        label_id = label_map[example.label]

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens = ["<s>"] + tokens_a + ["</s>"]
        segment_ids = [0] * (len(tokens_a) + 2)
        tokens += tokens_b + ["</s>"]
        segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [1] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        segment_ids += padding
        #print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      priori=example.priori))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def eval_1(preds, labels):
    TP = ((preds == 1) & (labels == 1)).sum()
    FN = ((preds == 0) & (labels == 0)).sum()
    TN = ((preds == 0) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + FN + 0.001)
    success = TP + FN
    fail = TN + FP
    acc = success / (success + fail + 0.001)
    return TP, TN, FN, FP, precision, recall, success, fail, acc


def eval_2(mapping):
    success = 0
    fail = 0
    for idx in mapping.keys():
        similarity, prog_label, fact_label, gold_label = mapping[idx]
        if prog_label == fact_label:
            success += 1
        else:
            fail += 1
    acc = success / (success + fail + 0.001)

    return success, fail, acc


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    TP, TN, FN, FP, precision, recall, success, fail, acc = eval_1(preds, labels)
    result = {"TP": TP, "TN": TN, "FN": FN, "FP": FP,
              "precision": precision, "recall": recall, "success": success, "fail": fail, "acc": acc}

    return result


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataLoader(args, processor, tokenizer, phase=None):
    dataset_dict = {"train": args.train_set, "dev": args.dev_set, "std_test": args.std_test_set,
                    "complex_test": args.complex_test_set,
                    "small_test": args.small_test_set, "simple_test": args.simple_test_set}
    label_list, _ = processor.get_labels()

    examples = processor.get_examples(args.data_dir, dataset_dict[phase])
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    batch_size = args.train_batch_size if phase == "train" else args.eval_batch_size
    epoch_num = args.num_train_epochs if phase == "train" else 1
    num_optimization_steps = int(len(examples) / batch_size / args.gradient_accumulation_steps) * epoch_num
    logger.info("Examples#: {}, Batch size: {}".format(len(examples), batch_size * args.gradient_accumulation_steps))
    logger.info("Total num of steps#: {}, Total num of epoch#: {}".format(num_optimization_steps, epoch_num))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_priori = torch.tensor([f.priori for f in features], dtype=torch.float)

    all_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_priori)
    if args.do_train_eval:
        sampler = SequentialSampler(all_data)
    else:
        sampler = RandomSampler(all_data) if phase == "train" else SequentialSampler(all_data)
    dataloader = DataLoader(all_data, sampler=sampler, batch_size=batch_size)

    return dataloader, num_optimization_steps, examples


def save_model(model_to_save):
    save_model_dir = os.path.join(args.output_dir, 'saved_model')
    mkdir(save_model_dir)
    output_model_file = os.path.join(save_model_dir, WEIGHTS_NAME)
    # output_config_file = os.path.join(save_model_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file, _use_new_zipfile_serialization=False)
    # model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(save_model_dir)

def softmax(input,T=1):
    output = [np.exp(i/T) for i in input]
    output_sum = sum(output)
    final = [i/output_sum for i in output]
    return final

def get_priori_distribution(statement,table_name,T=1):
    score = [0.1,0.1,0.1,0.1,0.6]
    pos_tags = nltk.pos_tag(statement.split())
    count_score = predict_count_api_by_rule(statement, table_name)
    comp_score = 1.5*float(predict_minmax_api_by_rule(pos_tags, table_name))
    superlative_score = 1.5*float(predict_comp_api_by_rule(pos_tags, table_name))
    negate_score = 1.5*float(predict_negate_api_by_rule(statement,table_name))
    count_score *= 2
    score[0] += count_score
    score[1] += comp_score
    score[2] += superlative_score
    score[3] += negate_score
    score = softmax(score,T)
    return score

def is_count_number(num):
    return 0 <= num <= 100

def predict_count_api_by_rule(statement,table_name):
    statement_list = parse_fact(statement).split()
    tmp_count_score = 0
    if 'only' in statement_list:
        index = statement_list.index('only')
        if index != len(statement_list)-1 and statement_list[index+1].isdigit() and is_count_number(int(statement_list[index+1])):
            tmp_count_score += 0.8
            if index+2<len(statement_list) and statement_list[index+2] == 'more':
                tmp_count_score -= 0.8
    if 'time' in statement_list:
        index = statement_list.index('time')
        if index != 0 and statement_list[index-1].isdigit()and 'time as many' not in parse_fact(statement) and is_count_number(int(statement_list[index-1])):
            tmp_count_score += 1
    if 'of' in statement_list:
        index = statement_list.index('of')
        if index != 0 and statement_list[index-1].isdigit() and is_count_number(int(statement_list[index-1])):
            tmp_count_score += 0.8
    if 'in' in statement_list:
        index = statement_list.index('in')
        tmp_list = statement.split()
        date_statement = False
        for word in tmp_list[index+1:]:
            if word in months or word in months_abbr:
                date_statement = True
                break
        if index != len(statement_list)-1 and statement_list[index+1].isdigit() and \
                is_count_number(int(statement_list[index+1])) and not date_statement:
            tmp_count_score += 0.6
            if index+2<len(statement_list) and statement_list[index+2] == 'more':
                tmp_count_score -= 0.6
            if index>0 and statement_list[index-1] == 'result':
                tmp_count_score -= 0.6
    if 'on' in statement.split():
        tmp_list = statement.split()
        index = tmp_list.index('on')
        date_statement = False
        for word in tmp_list[index+1:]:
            if word in months or word in months_abbr:
                date_statement = True
                break
        
        if index != len(tmp_list)-1 and tmp_list[index+1].isdigit() and \
                is_count_number(int(tmp_list[index+1])) and not date_statement:
            tmp_count_score += 0.35
            if index+2 < len(tmp_list) and tmp_list[index+2] == 'more':
                tmp_count_score -= 0.35
    if 'for' in statement_list:
        index = statement_list.index('for')
        if index != len(statement_list) - 1 and statement_list[index + 1].isdigit() and \
                is_count_number(int(statement_list[index+1])):
            tmp_count_score += 0.8
            if index+2<len(statement_list) and statement_list[index+2] == 'more':
                tmp_count_score -= 0.8
    if 'with' in statement_list:
        index = statement_list.index('with')
        if index != len(statement_list) - 1 and statement_list[index + 1].isdigit() and \
                is_count_number(int(statement_list[index + 1])):
            tmp_count_score += 0.1
            if index+2<len(statement_list) and statement_list[index+2] == 'more':
                tmp_count_score -= 0.1
    if 'have more' in parse_fact(statement):
        tmp_list = statement.replace('have more', '[TOK]').split()
        if '[TOK]' in tmp_list:
            index = tmp_list.index('[TOK]')
            if index + 1 < len(tmp_list) and '#' in tmp_list[index + 1]:
                entity = ''
                while True:
                    entity += tmp_list[index + 1]
                    if entity.count('#') == 2:
                        break
                    index += 1
                entity = entity.replace('#','').split(';')
                row_num = int(entity[1].split(',')[0])
                col_num = int(entity[1].split(',')[1])
                if row_num == 0:
                    with open('../data/all_csv/{}'.format(table_name),'r',encoding='utf8') as f:
                        t = pd.read_csv(f, delimiter="#")
                        if t.dtypes[t.columns[col_num]] == 'object':
                            tmp_count_score += 1
    if 'there be' in parse_fact(statement):
        tmp_list = parse_fact(statement).replace('there be','[TOK]').split()
        if '[TOK]' in tmp_list:
            index = tmp_list.index('[TOK]')
            if index+1 < len(tmp_list) and tmp_list[index + 1].isdigit() and is_count_number(int(tmp_list[index + 1])):
                tmp_count_score += 0.8
    if 'out of' in parse_fact(statement):
        tmp_list = parse_fact(statement).replace('out of','[TOK]').split()
        if '[TOK]' in tmp_list:
            index = tmp_list.index('[TOK]')
            if index+1 < len(tmp_list) and tmp_list[index + 1].isdigit() and is_count_number(int(tmp_list[index + 1])):
                tmp_count_score += 0.9
    if 'a total of' in parse_fact(statement):
        tmp_list = parse_fact(statement).replace('a total of','[TOK]').split()
        if '[TOK]' in tmp_list:
            index = tmp_list.index('[TOK]')
            if index+1 < len(tmp_list) and tmp_list[index + 1].isdigit() and is_count_number(int(tmp_list[index + 1])):
                tmp_count_score += 0.5
    if parse_fact(statement).split()[0].isdigit() and is_count_number(int(parse_fact(statement).split()[0])) and '-' not in parse_fact(statement):
        tmp_count_score += 0.7
    if tmp_count_score>0:
        return tmp_count_score
    else:
        return tmp_count_score
def check_if_word_in_table(table_name, word):
    table = pd.read_csv(os.path.join('../data/all_csv',table_name),sep='#')
    flag = False
    for col, item in table.iteritems():
        if word in col.split():
            flag = True
            break
        if item.dtypes == 'int64' or item.dtypes == 'float64':
            continue
        for wd in item:
            if word in str(wd).split():
                flag = True
                break
        if flag:
            break
    return flag
    
def predict_minmax_api_by_rule(pos_tags, table_name):
    
    minmax_tags = ['JJS','RBS']
    word_pond = ['minimum']
    flag = False
    for word, pos in pos_tags:
        if pos in minmax_tags or word in word_pond:
            flag = not check_if_word_in_table(table_name, word)
            if flag:
                break
    return flag

def predict_comp_api_by_rule(pos_tags, table_name):
    comp_tags = ['JJR','RBR']
    flag = False
    for word, pos in pos_tags:
        if pos in comp_tags:
            flag = flag = not check_if_word_in_table(table_name, word)
            if flag:
                break
    return flag

def predict_negate_api_by_rule(statement, table_name):
    word_pond = ['not', 'no', 'never', "didn't", "won't", "wasn't", "isn't", "aren't",
                          "haven't", "weren't", "won't", 'neither', 'none', 'unable', 'outside']
    flag = False
    for word in statement.split():
        if word in word_pond:
            flag = not check_if_word_in_table(table_name, word)
            if flag:
                break
    return flag

def predict_rank_api_by_rule(statement, table_name):
    word_pond = ['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth',
                '1st', '2nd', '3rd','4th','5th','6th','7th','8th','9th','10th','last']
    flag = False
    for word in statement.split():
        if word in word_pond:
            flag = not check_if_word_in_table(table_name, word)
            if flag:
                break
    return flag


def run_train(device, processor, tokenizer, model, writer, phase="train"):
    logger.info("\n************ Start Training *************")

    tr_dataloader, tr_num_steps, tr_examples = get_dataLoader(args, processor, tokenizer, phase="train")

    model.train()

    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = \
        [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=tr_num_steps)
    optimizer.zero_grad()

    global_step = 0
    best_acc = 0.0
    n_gpu = torch.cuda.device_count()

    for ep in trange(args.num_train_epochs, desc="Training"):
        for step, batch in tqdm(enumerate(tr_dataloader)):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, priori = batch
            logits, loss, final_out_logits, origin_gates = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
            guide_loss = loss_fct(torch.nn.functional.log_softmax(origin_gates, dim=1), priori)
            loss += args.lmd * guide_loss
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            writer.add_scalar('{}/loss'.format(phase), loss.item(), global_step)

            loss.backward()
            del loss

            if (step + 1) % args.gradient_accumulation_steps == 0:  # optimizer
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            model.eval()
            torch.set_grad_enabled(False)

            if args.do_eval and (((step + 1) % args.gradient_accumulation_steps == 0 and global_step % args.period == 0) or (ep==0 and step==0)):
                model_to_save = model.module if hasattr(model, 'module') else model

                dev_acc = run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=True,
                                   phase="dev")
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logger.info(">> Save model. Best acc: {:.4}. Epoch {}".format(best_acc, ep))
                    save_model(model_to_save)  # save model
                    logger.info(">> Now the best acc is {:.4}\n".format(dev_acc))

            model.train()
            torch.set_grad_enabled(True)

    return global_step


def run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=False,
             phase=None):
    sys.stdout.flush()
    logger.info("\n************ Start {} *************".format(phase))

    model.eval()

    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
    cross_entropy = nn.CrossEntropyLoss(reduction='none')

    dataloader, num_steps, examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    eval_loss = 0.0
    eval_guide_loss = 0.0
    num_steps = 0
    preds = []
    preds_0, preds_1, preds_2, preds_3, preds_4 = [],[],[],[],[]
    all_labels = []
    mapping = []
    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids, priori = batch
        num_steps += 1

        with torch.no_grad():

            logits, tmp_loss, final_out_logits, origin_gates = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
            guide_loss = loss_fct(torch.nn.functional.log_softmax(origin_gates, dim=1), priori)

            eval_loss += tmp_loss.mean().item()
            eval_guide_loss += guide_loss.mean().item()
            logits_sigmoid = final_out_logits
            loss = []
            for l in logits:
                loss.append(cross_entropy(l.squeeze(1), label_ids.view(-1)).view(-1,1))
            if len(loss) == 1:
                loss_mat = loss[0].view(-1,1)
            else:
                loss_mat = torch.cat(loss, dim=1) # bsz * # of experts
            logits_sigmoid_0 = torch.nn.functional.softmax(logits[0].squeeze(1), dim=1)
            logits_sigmoid_1 = torch.nn.functional.softmax(logits[1].squeeze(1), dim=1)
            logits_sigmoid_2 = torch.nn.functional.softmax(logits[2].squeeze(1), dim=1)
            logits_sigmoid_3 = torch.nn.functional.softmax(logits[3].squeeze(1), dim=1)
            logits_sigmoid_4 = torch.nn.functional.softmax(logits[4].squeeze(1), dim=1)
            if len(preds) == 0:
                preds.append(logits_sigmoid.detach().cpu().numpy())
                preds_0.append(logits_sigmoid_0.detach().cpu().numpy())
                preds_1.append(logits_sigmoid_1.detach().cpu().numpy())
                preds_2.append(logits_sigmoid_2.detach().cpu().numpy())
                preds_3.append(logits_sigmoid_3.detach().cpu().numpy())
                preds_4.append(logits_sigmoid_4.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits_sigmoid.detach().cpu().numpy(), axis=0)
                preds_0[0] = np.append(preds_0[0], logits_sigmoid_0.detach().cpu().numpy(), axis=0)
                preds_1[0] = np.append(preds_1[0], logits_sigmoid_1.detach().cpu().numpy(), axis=0)
                preds_2[0] = np.append(preds_2[0], logits_sigmoid_2.detach().cpu().numpy(), axis=0)
                preds_3[0] = np.append(preds_3[0], logits_sigmoid_3.detach().cpu().numpy(), axis=0)
                preds_4[0] = np.append(preds_4[0], logits_sigmoid_4.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()

            start = step * args.eval_batch_size if not args.do_train_eval else step * args.train_batch_size
            end = start + len(labels)
            batch_range = list(range(start, end))

            table_name = [examples[i].table_name for i in batch_range]
            labels = label_ids.detach().cpu().numpy().tolist()
            all_labels.extend(labels)
            loss_mat_cpu = loss_mat.detach().cpu().numpy().tolist()
            for i, t_name in enumerate(table_name):
                mapping.append([str(loss_mat_cpu[i][0]), str(loss_mat_cpu[i][1]), str(loss_mat_cpu[i][2]), str(loss_mat_cpu[i][3]), str(loss_mat_cpu[i][4])])

    result = {}
    result['acc'] = 0
    eval_loss /= num_steps
    eval_guide_loss /= num_steps
    preds = np.argmax(preds[0], axis=1)
    preds_0 = np.argmax(preds_0[0], axis=1)
    preds_1 = np.argmax(preds_1[0], axis=1)
    preds_2 = np.argmax(preds_2[0], axis=1)
    preds_3 = np.argmax(preds_3[0], axis=1)
    preds_4 = np.argmax(preds_4[0], axis=1)
    pred_for_test, label_for_test = [] ,[]
    for pred, label in zip(preds,all_labels):
        pred_for_test.append(pred)
        label_for_test.append(label)
            
    result = compute_metrics(np.asarray(pred_for_test), np.asarray(label_for_test))
    result_0 = compute_metrics(np.asarray(preds_0), np.asarray(all_labels))
    result_1 = compute_metrics(np.asarray(preds_1), np.asarray(all_labels))
    result_2 = compute_metrics(np.asarray(preds_2), np.asarray(all_labels))
    result_3 = compute_metrics(np.asarray(preds_3), np.asarray(all_labels))
    result_4 = compute_metrics(np.asarray(preds_4), np.asarray(all_labels))
    result['acc_0'] = result_0['acc']
    result['acc_1'] = result_1['acc']
    result['acc_2'] = result_2['acc']
    result['acc_3'] = result_3['acc']
    result['acc_4'] = result_4['acc']
    result['{}_loss'.format(phase)] = eval_loss
    result['{}_guide_loss'.format(phase)] = eval_guide_loss
    result['global_step'] = global_step
    logger.info(result)
    if tensorboard and writer is not None:
        for key in sorted(result.keys()):
            writer.add_scalar('{}/{}'.format(phase, key), result[key], global_step)
    json.dump(mapping, open('./{}_moe_roberta_lmd_0.1.json'.format(phase),'w', encoding='utf8'))
        
    model.train()
    return result['acc']


def main():
    mkdir(args.output_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))
    cache_dir = args.cache_dir

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    save_code_log_path = args.output_dir

    logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %H:%M', level=logging.INFO,
                        handlers=[logging.FileHandler("{0}/{1}.log".format(save_code_log_path, 'output')),
                                  logging.StreamHandler()])
    logger.info(args)
    logger.info("Command is: %s" % ' '.join(sys.argv))
    logger.info("Device: {}, n_GPU: {}".format(device, n_gpu))
    logger.info("Datasets are loaded from {}\nOutputs will be saved to {}\n".format(args.data_dir, args.output_dir))

    processor = DataProcessor()

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    load_dir = args.load_dir if args.load_dir else args.bert_model
    logger.info('Model is loaded from %s' % load_dir)
    label_list = processor.get_labels()
    config = RobertaConfig.from_json_file(os.path.join(args.bert_model,'config.json'))
    model = RobertaMoEForSequenceClassification(config, num_public_layers=12, num_experts=5,num_labels=2, num_gate_layer=2)
    model.load_roberta(args.bert_model)
    if args.load_dir:
        model.load_state_dict(torch.load(load_dir+'/pytorch_model.bin'))
        print('parameters loaded successfully.')
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model,device_ids=[0, 1])

    if args.do_train:
        run_train(device, processor, tokenizer, model, writer, phase="train")

    if args.do_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="dev")
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="std_test")

    if args.do_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="std_test")

    if args.do_train_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="train")

    if args.do_complex_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="complex_test")

    if args.do_small_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="small_test")

    if args.do_simple_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="simple_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_train_eval", action='store_true')
    parser.add_argument("--add_unk", action='store_true')
    parser.add_argument("--do_simple_test", action='store_true')
    parser.add_argument("--do_complex_test", action='store_true')
    parser.add_argument("--do_small_test", action='store_true')
    parser.add_argument("--load_dir", help="load model checkpoints")
    parser.add_argument("--data_dir", help="path to data", default='../processed_datasets_revised/tsv_data_horizontal')
    parser.add_argument("--train_set", default="train")
    parser.add_argument("--dev_set", default="dev")
    parser.add_argument("--std_test_set", default="test")
    parser.add_argument("--small_test_set", default="small_test")
    parser.add_argument("--complex_test_set", default="complex_test")
    parser.add_argument("--simple_test_set", default="simple_test")
    parser.add_argument("--output_dir", default='./outputs_moe')
    parser.add_argument("--cache_dir", default="./roberta", type=str, help="store downloaded pre-trained models")
    parser.add_argument('--period', type=int, default=1000)
    parser.add_argument("--bert_model", default="../roberta_large", type=str)
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--task_name", default="LPA", type=str)
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--train_batch_size", default=32)
    parser.add_argument("--eval_batch_size", default=32)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20)
    parser.add_argument("--lmd",default=0.1, type=float, help="the ratio of guide loss in the ttl loss")
    parser.add_argument("--warmup_proportion", default=0.3, type=float, help="0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()
    main()
