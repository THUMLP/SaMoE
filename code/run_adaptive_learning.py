from __future__ import absolute_import, division, print_function
import os

from nltk.util import pr
from torch._C import dtype
from typing_extensions import final

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import random
import sys
import re
import json
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
import torch.nn.functional
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME
from transformers import RobertaTokenizer, RobertaConfig
from pytorch_pretrained_bert.optimization import BertAdam
from tensorboardX import SummaryWriter
from interactive_model import RobertaMoEForSequenceClassification_adaptive
import logging

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
    def __init__(self, table_name, text_a, text_b=None, label=None,expert_ability=None):
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
        self.expert_ability = expert_ability


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, expert_ability):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.expert_ability = expert_ability

def softmax_func(x, T=1):
    return np.exp(x/T)/np.sum(np.exp(x/T),axis=-1, keepdims=True)

class DataProcessor(object):
    def get_examples(self, data_dir, dataset=None):
        logger.info('Get examples from: {}.tsv'.format(dataset))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset)), dataset))

    def get_labels(self):
        return [0, 1], len([0, 1])

    def _read_tsv(cls, input_file, dataset):
        if dataset == 'test':
            dataset = 'std_test'
        try:
            loss_info = json.load(open('{}_moe_roberta_lmd_0.1.json'.format(dataset),'r'))
        except Exception:
            loss_info = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for idx, line in enumerate(f):
                entry = line.strip().split('\t')
                table_name = entry[0]
                table_str = parse_fact(entry[3])
                statement = entry[4]
                label = int(entry[5])
                #print(idx)
                if len(loss_info) == 0:
                    expert_loss = [0.1,0.1,0.1,0.1,0.1] # for test/val, without any loss information
                else:
                    expert_loss = [float(one) for one in loss_info[idx]] # for train
                pre_var = np.var(expert_loss)
                lmd = np.sqrt(args.target_variance / pre_var)
                expert_ability = softmax_func(-lmd * np.asarray(expert_loss), T=1)
                lines.append([table_name, statement, table_str, label, expert_ability])
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
            expert_ability = line[4]
            examples.append((InputExample(table_name=table_name, text_a=text_a, text_b=text_b, label=label, expert_ability=expert_ability)))
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
                                      expert_ability=example.expert_ability))
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
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_expert_ability = torch.tensor([f.expert_ability for f in features], dtype=torch.float)

    all_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_expert_ability)
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

def run_train(device, processor, tokenizer, model, writer, phase="train"):
    logger.info("\n************ Start Training *************")

    tr_dataloader, tr_num_steps, tr_examples = get_dataLoader(args, processor, tokenizer, phase="train")

    model.train()

    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    train_param_name = [n for n, p in param_optimizer if (n.startswith('roberta.encoder.shift_layer') or n.startswith('roberta.encoder.shift_classifier'))]
    optimizer_grouped_parameters = \
        [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n in train_param_name], 'weight_decay': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and n in train_param_name], 'weight_decay': 0.0}]
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
            input_ids, input_mask, label_ids, expert_ability = batch
            _, _, _, shifted_gates = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)

            adaptive_loss = loss_fct(torch.nn.functional.log_softmax(shifted_gates, dim=1), expert_ability)
            loss = adaptive_loss
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
            if global_step == args.early_stop_step:
                return global_step

    return global_step

def run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=False,
             phase=None):
    sys.stdout.flush()
    logger.info("\n************ Start {} *************".format(phase))

    model.eval()

    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')

    dataloader, num_steps, examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    eval_loss = 0.0
    eval_adaptive_loss = 0.0
    num_steps = 0
    preds = []
    preds_0, preds_1, preds_2, preds_3, preds_4 = [],[],[],[],[]
    all_labels = []
    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids, expert_ability = batch
        num_steps += 1

        with torch.no_grad():

            logits, tmp_loss, shifted_final_logits, shifted_gates = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)

            adaptive_loss = loss_fct(torch.nn.functional.log_softmax(shifted_gates, dim=1), expert_ability)

            eval_loss += tmp_loss.mean().item()
            eval_adaptive_loss += adaptive_loss.mean().item()
            logits_sigmoid = shifted_final_logits
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
            all_labels.extend(labels)


    result = {}
    result['acc'] = 0
    eval_loss /= num_steps
    eval_adaptive_loss /= num_steps
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
    result['{}_adaptive_loss'.format(phase)] = eval_adaptive_loss
    result['global_step'] = global_step
    logger.info(result)
    if tensorboard and writer is not None:
        for key in sorted(result.keys()):
            writer.add_scalar('{}/{}'.format(phase, key), result[key], global_step)
        
    model.train()
    return result['acc']


def main():
    mkdir(args.output_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))

    # device = torch.device("cuda")
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

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    load_dir = args.load_dir if args.load_dir else args.bert_model
    logger.info('Model is loaded from %s' % load_dir)
    config = RobertaConfig.from_json_file(os.path.join(args.bert_model,'config.json'))

    model = RobertaMoEForSequenceClassification_adaptive(config, num_public_layers=12, num_experts=5,num_labels=2, num_gate_layer=2)

    model.load_roberta(args.bert_model)
    if args.load_dir:
        try:
            model.load_state_dict(torch.load(load_dir+'/pytorch_model.bin'))
            print('parameters loaded successfully.')
        except Exception:
            model.load_MoE(load_dir)
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
    parser.add_argument("--output_dir", default='./outputs_adaptive_learning')
    parser.add_argument('--period', type=int, default=1000)
    parser.add_argument("--bert_model", default="../roberta_large", type=str,
                        help="list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, "
                             "bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--train_batch_size", default=32)
    parser.add_argument("--eval_batch_size", default=32)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20)
    parser.add_argument("--lmd",default=0.1, type=float, help="the ratio of guide loss in the ttl loss")
    parser.add_argument("--warmup_proportion", default=0.3, type=float, help="0.1 = 10%% of training.")
    parser.add_argument("--target_variance", default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument("--early_stop_step", help="step to stop the adaptive learning", default=5000)
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()
    main()

