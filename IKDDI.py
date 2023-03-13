
import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from gensim.models import KeyedVectors

import linecache
import re
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize
import torch.optim as optim
import math
from sklearn import metrics
from sklearn.metrics import accuracy_score # sklearn中的精准率

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig
import os
import random

import xlrd
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

config_path = 'BioBERT_pytorch/config.json'
model_path = 'BioBERT_pytorch'
vocab_path = 'BioBERT_pytorch/vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_path)

PAD, CLS = '[PAD]', '[CLS]'
sequence_length = 128
BATCH_SIZE = 32

sentences = []
sentences_id = []
label = []
atten_masks = []

e1_pos = []
e2_pos = []
dist_ra_e1 = []
dist_ra_e2 = []

article_num_list = []
sentence_num_list = []

text_DB = []

e1_DB_text_id = []
e2_DB_text_id = []
atten_masks_DB_e1 = []
atten_masks_DB_e2 = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def attention_masks(ids):
    id_mask = [int(i > 0) for i in ids]  # PAD: 0; or: 1
    return id_mask

def load_DBText(excel_name):
    info_DB = []
    workbook = xlrd.open_workbook(excel_name)
    sheet_1 = workbook.sheet_by_name('Sheet 1')
    nrows = sheet_1.nrows

    for i in range(0, nrows):
        info_curr = []
        drug1_text = sheet_1.cell(i, 0).value
        drug2_text = sheet_1.cell(i, 1).value
        info_curr.append(drug1_text)
        info_curr.append(drug2_text)
        info_DB.append(info_curr)

    return info_DB

def pos(x):
    if x < -sequence_length:
        return 0
    if sequence_length >= x >= -sequence_length:
        return x + sequence_length + 1
    if x > sequence_length:
        return (sequence_length + 1) * 2

def textToId(text):
    if text == 'none':
        input_ids = [0] * sequence_length
        input_mask = [0] * sequence_length
    else:
        sen_curr = text.lower()
        dicts = {x: ' ' for x in punctuation}
        punc_table = str.maketrans(dicts)
        sen_curr = sen_curr.translate(punc_table)
        tokens = tokenizer.tokenize(sen_curr)
        len_tokens = len(tokens)
        input_mask = [1]*len_tokens
        if sequence_length > len_tokens:
            for j in range(sequence_length - len_tokens):
                tokens.append(PAD)
                input_mask.append(0)
        elif sequence_length < len_tokens:
            for k in range(len_tokens - sequence_length):
                del (tokens[-1])
                del (input_mask[-1])

        input_ids = tokenizer.encode(tokens)

    return input_ids, input_mask

def load_data(file):
    total_line = len(open(file).readlines())

    for i in range(total_line):
        line = linecache.getline(file, i + 1)
        line = line.strip()
        list_1 = [j.start() for j in re.finditer('=', line)]
        list_2 = [j.start() for j in re.finditer('\$', line)]
        type = line[list_1[2] + 1: list_2[2]]

        if type.lower() == 'false':
            label.append(0.0)
        elif type.lower() == 'effect':
            label.append(1.0)
        elif type.lower() == 'advise':
            label.append(2.0)
        elif type.lower() == 'mechanism':
            label.append(3.0)
        elif type.lower() == 'int':
            label.append(4.0)
        else:
            label.append(0.0)

        article_num = int(line[list_1[-2] + 1: list_2[-2]])
        sentence_num = int(line[list_1[-1] + 1: list_2[-1]])

        article_temp = []
        article_temp.append(article_num)
        article_num_list.append(article_temp * 128)

        sentence_temp = []
        sentence_temp.append(sentence_num)
        sentence_num_list.append(sentence_temp * 128)

        sentence_raw = line[list_1[3] + 1: list_2[3]]
        sentence_raw = sentence_raw.lower()

        dicts = {x: ' ' for x in punctuation}
        punc_table = str.maketrans(dicts)
        sentence_raw = sentence_raw.translate(punc_table)
        sentence_split = nltk.word_tokenize(sentence_raw)
        sentence_split_new = [x for x in sentence_split if x != '']
        sentence_content = ' '.join(sentence_split_new)

        token = tokenizer.tokenize(sentence_content)
        sentences.append(token)

        pos_1 = sentence_split_new.index('e1')
        pos_2 = sentence_split_new.index('e2')

        e1_pos.append(pos_1)
        e2_pos.append(pos_2)

        dis_curr_line_1 = []
        dis_curr_line_2 = []
        for j in range(0, sequence_length):
            e_r_1 = j - pos_1
            e_r_2 = j - pos_2
            dis_curr_line_1.append(pos(e_r_1))
            dis_curr_line_2.append(pos(e_r_2))
        dist_ra_e1.append(dis_curr_line_1)
        dist_ra_e2.append(dis_curr_line_2)

        input_ids = tokenizer.encode(token)
        len_ids = len(input_ids)

        if sequence_length > len_ids:
            for j in range(sequence_length - len_ids):
                input_ids.append(0)
        elif sequence_length < len_ids:
            for k in range(len_ids - sequence_length):
                del(input_ids[-1])

        sentences_id.append(input_ids)

        input_mask = attention_masks(input_ids)
        atten_masks.append(input_mask)

        drug1_text = text_DB[i][0]
        drug2_text = text_DB[i][1]
        drug1_text_id, drug1_text_mask = textToId(drug1_text)
        drug2_text_id, drug2_text_mask = textToId(drug2_text)

        e1_DB_text_id.append(drug1_text_id)
        e2_DB_text_id.append(drug2_text_id)
        atten_masks_DB_e1.append(drug1_text_mask)
        atten_masks_DB_e2.append(drug2_text_mask)

def gen_dataloader():
    for i in range(0, len(sentences_id)):
        if len(sentences_id[i]) > sequence_length:
            for k in range(len(sentences_id[i]) - sequence_length):
                del(sentences_id[i][-1])
                del(atten_masks[i][-1])

    for i in range(0, len(e1_DB_text_id)):
        if len(e1_DB_text_id[i]) > sequence_length:
            for k in range(len(e1_DB_text_id[i]) - sequence_length):
                del(e1_DB_text_id[i][-1])

    for i in range(0, len(atten_masks_DB_e1)):
        if len(atten_masks_DB_e1[i]) > sequence_length:
            for k in range(len(atten_masks_DB_e1[i]) - sequence_length):
                del(atten_masks_DB_e1[i][-1])

    for i in range(0, len(e2_DB_text_id)):
        if len(e2_DB_text_id[i]) > sequence_length:
            for k in range(len(e2_DB_text_id[i]) - sequence_length):
                del(e2_DB_text_id[i][-1])

    for i in range(0, len(atten_masks_DB_e2)):
        if len(atten_masks_DB_e2[i]) > sequence_length:
            for k in range(len(atten_masks_DB_e2[i]) - sequence_length):
                del(atten_masks_DB_e2[i][-1])

    all_index_ten = torch.LongTensor(sentences_id)
    dist_ra_e1_ten = torch.LongTensor(dist_ra_e1)
    dist_ra_e2_ten = torch.LongTensor(dist_ra_e2)

    article_num_ten = torch.LongTensor(article_num_list)
    sentence_num_ten = torch.LongTensor(sentence_num_list)

    real_input = torch.cat((all_index_ten, dist_ra_e1_ten, dist_ra_e2_ten, article_num_ten, sentence_num_ten), axis=1)
    e1_desc_ten = torch.LongTensor(e1_DB_text_id)
    e2_desc_ten = torch.LongTensor(e2_DB_text_id)
    e1_mask_ten = torch.LongTensor(atten_masks_DB_e1)
    e2_mask_ten = torch.LongTensor(atten_masks_DB_e2)
    attention_tokens = torch.LongTensor(atten_masks)
    real_target = torch.LongTensor(label)

    torch_dataset = Data.TensorDataset(real_input, attention_tokens, real_target, e1_desc_ten, e2_desc_ten,
                                       e1_mask_ten, e2_mask_ten)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return loader

def data_unpack(cat_data, N):
    list_x = torch.split(cat_data, [N, N, N, N, N], 1)
    x = list_x[0]
    e_r_1 = list_x[1]
    e_r_2 = list_x[2]
    article_lable = list_x[3]
    sentence_lable = list_x[4]
    return x, e_r_1, e_r_2, article_lable, sentence_lable

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dist_embedding = nn.Embedding(1000, 10)
        self.dist_embedding.weight.data.uniform_(-1e-3, 1e-3)
        kernel_sizes = [5]

        self.embedding_siz_DB = 1536
        self.out_channel_DB = 128
        self.conv_list_text = nn.ModuleList([nn.Conv1d(self.embedding_siz_DB,
                                                  self.out_channel_DB, w, padding=(w - 1) // 2) for w in kernel_sizes])
        self.hidden_size = 404
        dropout_1 = 0.5
        dropout_2 = 0.1
        embedding_dim = 808
        flag_class = 5
        hidden_size = 768
        out_A_len = 128

        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, bidirectional=True)
        self.conv_list = nn.ModuleList([nn.Conv1d(self.hidden_size * 2,
                                                  hidden_size, w, padding=(w - 1) // 2) for w in kernel_sizes])
        self.classifier = nn.Linear(len(kernel_sizes) * hidden_size + out_A_len, flag_class)
        self.dropout_1 = nn.Dropout(dropout_1)
        self.dropout_2 = nn.Dropout(dropout_2)

    def forward(self, x_in, e_r_1_in, e_r_2_in, article_lable_in, sentence_lable_in, masks, e1_text, e2_text, e1_text_mask, e2_text_mask):
        out_e1 = self.bert(e1_text, attention_mask=e1_text_mask)
        out_e1_cls = out_e1[1]
        out_e2 = self.bert(e2_text, attention_mask=e2_text_mask)
        out_e2_cls = out_e2[1]
        input_1 = torch.cat((out_e1_cls, out_e2_cls), 1)
        input_1 = input_1.unsqueeze(1)
        conv_outputs_1 = []
        for c in self.conv_list_text:
            conv_output_text = c(input_1.transpose(1, 2))
            conv_output_text = gelu(conv_output_text)
            conv_output_text, _ = torch.max(conv_output_text, -1)
            conv_outputs_1.append(conv_output_text)
        pooled_output_text = torch.cat(conv_outputs_1, 1)
        dist_1_emb = self.dist_embedding(e_r_1_in)
        dist_2_emb = self.dist_embedding(e_r_2_in)

        article_lable_emb = self.dist_embedding(article_lable_in)
        sentence_lable_emb = self.dist_embedding(sentence_lable_in)
        out = self.bert(x_in, attention_mask=masks)
        input_0 = out[0]

        x_cat = torch.cat((input_0, dist_1_emb, dist_2_emb, article_lable_emb, sentence_lable_emb), 2)
        input_1 = x_cat.permute(1, 0, 2)

        hidden_state = Variable(torch.zeros(2, len(x_in), self.hidden_size).cuda())
        cell_state = Variable(torch.zeros(2, len(x_in), self.hidden_size).cuda())

        output_1, (final_hidden_state, final_cell_state) = self.lstm(input_1, (hidden_state, cell_state))
        output_2 = output_1.permute(1, 0, 2)

        conv_outputs = []
        for c in self.conv_list:
            conv_output = c(output_2.transpose(1, 2))
            conv_output = gelu(conv_output)
            conv_output, _ = torch.max(conv_output, -1)
            conv_outputs.append(conv_output)
        pooled_output = torch.cat(conv_outputs, 1)

        pooled_output = self.dropout_2(pooled_output)

        output_cat = torch.cat((pooled_output, pooled_output_text), 1)

        logits = self.classifier(output_cat)

        return logits

if __name__ == '__main__':
    setup_seed(10)

    text_DB = load_DBText('train_text.xls')
    file_name = 'train_after.txt'
    load_data(file_name)
    loader = gen_dataloader()

    model = Model()
    model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, eps=1e-8)

    model.train()

    # Training
    for epoch in range(3):
        print('\nepoch: {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0
        sum_train = 0
        corrects_train = 0
        size_train = 0

        for i, (real_input, attention_tokens, real_target, e1_desc, e2_desc, e1_mask, e2_mask) in enumerate(loader):
            optimizer.zero_grad()

            real_target = real_target.cuda()
            real_input = real_input.cuda()
            attention_tokens = attention_tokens.cuda()
            e1_desc = e1_desc.cuda()
            e2_desc = e2_desc.cuda()
            e1_mask = e1_mask.cuda()
            e2_mask = e2_mask.cuda()

            x, e_r_1, e_r_2, article_lable, sentence_lable = data_unpack(real_input, sequence_length)
            out = model(x, e_r_1, e_r_2, article_lable, sentence_lable, attention_tokens, e1_desc, e2_desc, e1_mask, e2_mask)
            corrects_train_cur = (torch.max(out, 1)[1].view(real_target.size()) == real_target).sum()
            corrects_train += corrects_train_cur
            size_train = size_train + len(real_input)
            loss = criterion(out, real_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            sum_train += 1

        accuracy_train = 100.0 * corrects_train / size_train
        loss_average = running_loss / sum_train
        print('epoch %d' % (i), '结束，loss平均为：%f' % (running_loss / sum_train), 'acc = %f' % (accuracy_train))

    print("\n\nTotal number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    del loader
    del real_input
    del attention_tokens
    del corrects_train_cur
    del article_num_list
    del sentence_num_list

    # Test
    corrects, avg_loss = 0, 0
    corrects_test = 0
    sum_test = 0
    sentences = []
    sentences_id = []
    label = []
    atten_masks = []
    e1_pos = []
    e2_pos = []
    dist_ra_e1 = []
    dist_ra_e2 = []
    article_num_list = []
    sentence_num_list = []

    e1_DB_text_id = []
    e2_DB_text_id = []
    atten_masks_DB_e1 = []
    atten_masks_DB_e2 = []

    file_name = 'test_after.txt'
    load_data(file_name)
    loader = gen_dataloader()

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    for i, (input_test_unit, attention_test, target_test_unit, e1_desc_test, e2_desc_test, e1_mask_test, e2_mask_test) in enumerate(loader):
        target_test_unit = target_test_unit.cuda()
        input_test_unit = input_test_unit.cuda()
        attention_test = attention_test.cuda()
        e1_desc_test = e1_desc_test.cuda()
        e2_desc_test = e2_desc_test.cuda()
        e1_mask_test = e1_mask_test.cuda()
        e2_mask_test = e2_mask_test.cuda()

        x, e_r_1, e_r_2, article_lable, sentence_lable = data_unpack(input_test_unit, sequence_length)
        out_test = model(x, e_r_1, e_r_2, article_lable, sentence_lable, attention_test, e1_desc_test, e2_desc_test, e1_mask_test, e2_mask_test)
        if i == 0:
            out_test_matrix = out_test
            out_lable_matrix = target_test_unit
        elif i > 0:
            out_test_matrix = torch.cat((out_test_matrix, out_test), 0)
            out_lable_matrix = torch.cat((out_lable_matrix, target_test_unit), 0)

    y_pred = torch.max(out_test_matrix, 1)[1].view(out_lable_matrix.size())
    y_true = out_lable_matrix

    y_pred = y_pred.cuda().cpu()
    y_true = y_true.cuda().cpu()

    micro_p = precision_score(y_true, y_pred, average='micro')
    micro_r = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    print(' micro_p = ' + str(micro_p) + 'micro_r' + str(micro_r) + 'micro_f1' + str(micro_f1))

    macro_p = precision_score(y_true, y_pred, average='macro')
    macro_r = recall_score(y_true, y_pred, average='macro')
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)

    print(' macro_p = ' + str(macro_p) + 'macro_r' + str(macro_r) + 'macro_f1' + str(macro_f1))

    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

